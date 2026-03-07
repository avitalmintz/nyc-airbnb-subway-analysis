import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Airbnb & Subway Proximity",
    page_icon="🚇",
    layout="wide"
)

# ── Load data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/airbnb_with_subway_features.csv")
    subway = pd.read_csv("data/MTA_Subway_Stations.csv")
    borough_map = {"M": "Manhattan", "Bk": "Brooklyn", "Q": "Queens", "Bx": "Bronx", "SI": "Staten Island"}
    subway["Borough_Full"] = subway["Borough"].map(borough_map)
    subway_clean = subway.drop_duplicates(subset=["GTFS Latitude", "GTFS Longitude"])
    with open("data/nyc-borough.geojson", "r") as f:
        borough_geo = json.load(f)
    return df, subway_clean, borough_geo

df, subway_clean, borough_geo = load_data()

# ── Sidebar filters ─────────────────────────────────────────────────────────
st.sidebar.title("Filters")

boroughs = st.sidebar.multiselect(
    "Borough",
    options=sorted(df["neighbourhood_group"].unique()),
    default=sorted(df["neighbourhood_group"].unique())
)

room_types = st.sidebar.multiselect(
    "Room Type",
    options=sorted(df["room_type"].unique()),
    default=sorted(df["room_type"].unique())
)

price_range = st.sidebar.slider(
    "Price Range ($ / night)",
    min_value=int(df["price_capped"].min()),
    max_value=int(df["price_capped"].max()),
    value=(int(df["price_capped"].min()), int(df["price_capped"].max()))
)

# Apply filters
mask = (
    df["neighbourhood_group"].isin(boroughs)
    & df["room_type"].isin(room_types)
    & df["price_capped"].between(*price_range)
)
fdf = df[mask]

st.sidebar.markdown("---")
st.sidebar.metric("Listings shown", f"{len(fdf):,}", f"of {len(df):,} total")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: INTRO & OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
st.title("Is Airbnb Listing Price Associated with Subway Proximity in NYC?")
st.markdown("""
**DATA 227 — Data Visualization Project | Winter 2026**

---

New York City's subway network moves millions of people across the five boroughs, making transit access
a central part of how the city functions. For short-term rental guests, being close to reliable transit
often determines whether a stay feels convenient or frustrating — so examining this relationship can
clarify which neighborhoods offer the best balance of price and accessibility, and what prospective
Airbnb hosts should expect when setting prices. Ultimately, we want to ask: **what is the relationship
between rental price and transit access in NYC?**

To answer this question, we merged **{:,} Airbnb listings** with **{:,} MTA subway station locations**,
calculated each listing's distance to its nearest station, and counted how many stations fall within a
half-mile radius.
""".format(len(df), len(subway_clean)))

# Key metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Median Price", f"${fdf['price'].median():,.0f}")
col2.metric("Avg Stations (0.5 mi)", f"{fdf['stations_05mi'].mean():.1f}")
col3.metric("Listings", f"{len(fdf):,}")
col4.metric("Neighborhoods", f"{fdf['neighbourhood'].nunique()}")

# ── Interactive Map ──────────────────────────────────────────────────────────
st.markdown("## Interactive Map")
st.markdown("""
Each dot below is an Airbnb listing, colored from yellow (cheaper) to red (expensive). Blue dots mark
subway station entrances. Use the sidebar filters to explore specific boroughs, room types, or price
ranges. Whether the spatial overlap between dense subway coverage and high prices reflects genuine
transit-driven demand — or simply the broader economic premium of certain neighborhoods — is the
question this analysis seeks to answer.
""")

map_sample = fdf.sample(n=min(5000, len(fdf)), random_state=42) if len(fdf) > 5000 else fdf

fig_map = px.scatter_mapbox(
    map_sample,
    lat="latitude", lon="longitude",
    color="price_capped",
    color_continuous_scale="YlOrRd",
    opacity=0.5,
    hover_name="name",
    hover_data={
        "price": ":$.0f",
        "neighbourhood_group": True,
        "room_type": True,
        "stations_05mi": True,
        "latitude": False, "longitude": False, "price_capped": False
    },
    labels={"price_capped": "Price ($)", "stations_05mi": "Stations (0.5mi)"},
)
fig_map.add_trace(go.Scattermapbox(
    lat=subway_clean["GTFS Latitude"],
    lon=subway_clean["GTFS Longitude"],
    mode="markers",
    marker=dict(size=5, color="#1E88E5", opacity=0.7),
    name="Subway Stations",
    text=subway_clean["Stop Name"],
    hoverinfo="text"
))
fig_map.update_layout(
    mapbox_style="carto-positron",
    mapbox_center={"lat": 40.7128, "lon": -74.0060},
    mapbox_zoom=10,
    height=600,
    margin=dict(l=0, r=0, t=0, b=0)
)
st.plotly_chart(fig_map, use_container_width=True)
st.caption("Figure 1: Airbnb listings colored by capped nightly price (YlOrRd scale) overlaid with MTA subway station locations (blue). Hover over any listing for price, borough, room type, and nearby station count. Up to 5,000 listings sampled when filters return a larger dataset.")
st.markdown("Manhattan listings, clustered around the densest subway network, also skew the most expensive. This spatial overlap is the core tension we will unpack throughout this analysis.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION: DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## Data Overview")
st.markdown("""
Before testing the relationship between subway access and price, we need to understand the data itself.
NYC's five boroughs differ dramatically in price, room type, and transit infrastructure. These baseline
differences will matter a great deal when we interpret the subway proximity results, so let's establish
them first.
""")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### Price Distribution by Borough")
    st.markdown("Box plots are well-suited here because they expose the median, interquartile range, and outliers simultaneously for a continuous variable across categories — giving a richer picture than a bar of means alone.")
    fig_box = px.box(
        fdf, x="neighbourhood_group", y="price_capped",
        color="neighbourhood_group",
        category_orders={"neighbourhood_group": fdf.groupby("neighbourhood_group")["price_capped"]
                         .median().sort_values(ascending=False).index.tolist()},
        labels={"price_capped": "Price ($/night)", "neighbourhood_group": "Borough"},
    )
    fig_box.update_layout(showlegend=False, height=450)
    st.plotly_chart(fig_box, use_container_width=True)
    st.caption("Figure 2: Nightly price distribution by borough. Boroughs ordered by median price (descending). Price capped at the 99th percentile to reduce outlier distortion.")
    st.markdown("Manhattan commands far higher nightly prices than the other boroughs, with a wider spread of outliers. Brooklyn is a distant second. Staten Island and the Bronx occupy the lower end of the price range.")

with col_b:
    st.markdown("### Room Type Composition by Borough")
    st.markdown("A stacked bar reveals both absolute listing volume and proportional room-type mix simultaneously, making cross-borough comparisons easier than two separate charts.")
    room_counts = fdf.groupby(["neighbourhood_group", "room_type"]).size().reset_index(name="count")
    fig_room = px.bar(
        room_counts, x="neighbourhood_group", y="count", color="room_type",
        labels={"count": "Listings", "neighbourhood_group": "Borough", "room_type": "Room Type"},
        barmode="stack"
    )
    fig_room.update_layout(height=450)
    st.plotly_chart(fig_room, use_container_width=True)
    st.caption("Figure 3: Listing count by borough, stacked by room type.")
    st.markdown("Room type varies greatly by borough. Manhattan and Brooklyn skew heavily toward entire home listings, which demand premium prices. This difference is one reason we can't simply compare raw prices across boroughs; we need to account for what's actually being rented.")
    st.markdown("Now that we understand how listings are distributed by price and type, the next question is: how does subway infrastructure compare across these same boroughs? If Manhattan has both the highest prices and the most subway stations, separating those two effects will be the central challenge of our analysis.")

# Subway density
col_c, col_d = st.columns(2)

with col_c:
    st.markdown("### Subway Stations by Borough")
    st.markdown("This chart establishes the supply side of the equation: how is MTA infrastructure distributed across the five boroughs?")
    station_by_borough = subway_clean["Borough_Full"].value_counts().reset_index()
    station_by_borough.columns = ["Borough", "Stations"]
    fig_stn = px.bar(station_by_borough, x="Borough", y="Stations",
                     color="Borough", labels={"Stations": "Station Locations"})
    fig_stn.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_stn, use_container_width=True)
    st.caption("Figure 4: Total MTA subway station locations by borough.")
    st.markdown("Manhattan has far more stations than any other borough, followed by Brooklyn and Queens. Staten Island has almost none. This asymmetry sets up a critical confound: any raw correlation between subway proximity and price may partly reflect the Manhattan premium rather than transit access itself.")

with col_d:
    st.markdown("### Avg Stations within 0.5 mi per Listing")
    st.markdown("Rather than counting stations in a borough, this metric normalizes to the listing level — how many stations can a typical guest in each borough walk to within half a mile?")
    avg_stn = fdf.groupby("neighbourhood_group")["stations_05mi"].mean().reset_index()
    avg_stn.columns = ["Borough", "Avg Stations"]
    fig_avg = px.bar(avg_stn.sort_values("Avg Stations", ascending=False),
                     x="Borough", y="Avg Stations", color="Borough")
    fig_avg.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_avg, use_container_width=True)
    st.caption("Figure 5: Average number of subway stations within 0.5 miles of listings in each borough.")
    st.markdown("Manhattan listings sit near far more stations on average, while outer-borough listings often have just one or two options within walking distance, or none at all.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION: CORE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## Core Analysis: Price vs. Subway Proximity")
st.markdown("""
The data overview revealed a potential confound: Manhattan has both the highest prices and the densest
subway network. So, when we see a positive correlation between station count and price, is that subway
access doing the work — or just borough location? Below we examine this at multiple levels — overall,
by borough, and by neighborhood.
""")

# Binned bar chart
st.markdown("### Average Price by Number of Nearby Stations (0.5 mi)")
st.markdown("Binning the continuous station count into ordered groups lets us see whether the price-proximity relationship is monotonic or whether there is a threshold beyond which additional stations no longer matter. The sample size label on each bar (n=) signals where the data are thin.")
bin_stats = fdf.groupby("station_bin", observed=True).agg(
    mean_price=("price_capped", "mean"),
    median_price=("price_capped", "median"),
    count=("price_capped", "count")
).reset_index()

st.markdown("*Toggle between median and mean below to see how sensitive the trend is to outliers.*")
metric_choice = st.radio("Metric", ["Median", "Mean"], horizontal=True, key="metric_radio")
price_col = "median_price" if metric_choice == "Median" else "mean_price"

fig_bins = px.bar(
    bin_stats, x="station_bin", y=price_col,
    text="count",
    labels={"station_bin": "Stations within 0.5 mi", price_col: f"{metric_choice} Price ($)",
            "count": "Listings"},
    color=price_col, color_continuous_scale="YlOrRd"
)
fig_bins.update_traces(texttemplate="n=%{text:,}", textposition="outside")
fig_bins.update_layout(height=450)
st.plotly_chart(fig_bins, use_container_width=True)
st.caption("Figure 6: Median or mean nightly price by binned count of subway stations within 0.5 miles. Bar labels show the number of listings in each bin. Toggle between Median and Mean using the radio button above.")
st.markdown("A general upward trend is visible — listings with more nearby stations tend to fetch higher prices — but the pattern is not perfectly monotonic and thins out considerably in the highest station-count bins. Critically, this raw picture does not control for borough: Manhattan listings, which happen to have both high prices and many nearby stations, are pulling the higher bins upward.")

# By borough facet
st.markdown("### Price vs. Stations — By Borough")
st.markdown(
    "Stratifying by borough removes the largest confound identified above — the Manhattan effect. "
    "If subway proximity genuinely drives price, we should see an upward slope *within* each borough "
    "independently, not just in the pooled data. Points with fewer than 10 listings are excluded to "
    "limit noise from sparse station-count cells."
)

borough_means = fdf.groupby(["neighbourhood_group", "stations_05mi"]).agg(
    mean_price=("price_capped", "mean"),
    count=("price_capped", "count")
).reset_index()
# Only show station counts with enough data
borough_means = borough_means[borough_means["count"] >= 10]

fig_facet = px.line(
    borough_means, x="stations_05mi", y="mean_price",
    color="neighbourhood_group", markers=True,
    labels={"stations_05mi": "Stations within 0.5 mi", "mean_price": "Mean Price ($)",
            "neighbourhood_group": "Borough"},
    title="Mean Price by Station Count (min 10 listings per point)"
)
fig_facet.update_layout(height=500)
st.plotly_chart(fig_facet, use_container_width=True)
st.caption("Figure 7: Mean nightly price vs. number of subway stations within 0.5 miles, broken out by borough. Only station counts with at least 10 listings are plotted.")
st.markdown("In some boroughs, the trend between price and station count weakens considerably, suggesting that much of what looked like a 'subway effect' in the previous chart was really just due to 3 of the 5 boroughs. Does access to more stations still move the needle within a given neighborhood context?")

# Distance to nearest station
st.markdown("### Price by Distance to Nearest Station")
st.markdown("Counting nearby stations measures density of access; distance to the *nearest* station measures isolation. If transit proximity is genuinely valued, we expect prices to decline as distance increases — the following box plot examines whether that hypothesis holds.")
fig_dist = px.box(
    fdf.dropna(subset=["dist_bin"]),
    x="dist_bin", y="price_capped",
    color="dist_bin",
    category_orders={"dist_bin": ["<0.1 mi", "0.1-0.25 mi", "0.25-0.5 mi", "0.5-1 mi", ">1 mi"]},
    labels={"dist_bin": "Distance to Nearest Station", "price_capped": "Price ($/night)"},
)
fig_dist.update_layout(showlegend=False, height=450)
st.plotly_chart(fig_dist, use_container_width=True)
st.caption("Figure 8: Distribution of nightly price by distance bin to the nearest subway station. Bins ordered from closest (<0.1 mi) to most isolated (>1 mi).")
st.markdown("From the data, we see that there is an effect from station distance and price, with listings closer to a station leading to a slightly more expensive rental on average. However, this relationship does not seem to be dramatic.")

# ── Correlation heatmap ──────────────────────────────────────────────────────
st.markdown("### Correlation Heatmap")
st.markdown("To see exactly how strong a relationship the two variables have, a correlation heatmap gives us precise values. Here, we examine pairwise correlations across all key numeric variables — this tells us not just whether stations correlate with price, but how that relationship compares to factors like bedroom count, availability, and ratings. We can also check whether the two subway variables are highly correlated with each other (which could cause issues in a regression). The RdBu_r diverging color scale makes direction immediately readable: blue = positive, red = negative, white ≈ zero.")
corr_cols = ["price_capped", "stations_05mi", "stations_1mi",
             "nearest_station_miles", "bedrooms", "beds", "rating",
             "number_of_reviews", "availability_365"]
corr_labels = ["Price", "Stations (0.5mi)", "Stations (1mi)",
               "Nearest Stn (mi)", "Bedrooms", "Beds", "Rating",
               "Reviews", "Availability"]
corr_matrix = fdf[corr_cols].corr().round(2)
corr_matrix.index = corr_labels
corr_matrix.columns = corr_labels

fig_corr = px.imshow(
    corr_matrix, text_auto=True, color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1, aspect="auto"
)
fig_corr.update_layout(height=550)
st.plotly_chart(fig_corr, use_container_width=True)
st.caption("Figure 9: Pearson correlation matrix across price, subway proximity, and listing attribute variables. Values range from -1 (strong negative) to +1 (strong positive).")
st.markdown("We see a positive correlation between nearby stations and price, but 0.21 and 0.25 are relatively low correlations — not the strongest indicator of price compared to the number of beds and bedrooms, which are roughly twice as impactful. Nearest station distance shows a negative correlation, indicating that the further a rental is from a station, the cheaper it tends to be. Notably, ratings and review counts sit near zero, suggesting guest-facing quality signals are not meaningfully priced into the market.")

# ── Neighborhood bubble chart ────────────────────────────────────────────────
st.markdown("### Neighborhood-Level View")
st.markdown("The charts so far have examined listings and boroughs in aggregate. This bubble chart zooms in on individual neighborhoods, with each bubble sized by the number of listings it contains. Neighborhoods with high station density but lower prices are particularly informative: they reveal cases where the subway story breaks down and other neighborhood characteristics likely dominate. Hover over any bubble to explore.")

nbhd = fdf.groupby("neighbourhood").agg(
    median_price=("price", "median"),
    mean_stations=("stations_05mi", "mean"),
    count=("price", "count"),
    borough=("neighbourhood_group", "first")
).reset_index()
nbhd = nbhd[nbhd["count"] >= 20]

fig_nbhd = px.scatter(
    nbhd, x="mean_stations", y="median_price",
    size="count", color="borough", hover_name="neighbourhood",
    labels={"mean_stations": "Avg Stations within 0.5 mi",
            "median_price": "Median Price ($)",
            "count": "Listings", "borough": "Borough"},
    size_max=40, opacity=0.7
)
fig_nbhd.update_layout(height=550)
st.plotly_chart(fig_nbhd, use_container_width=True)
st.caption("Figure 10: Neighborhood-level scatter plot of average subway stations within 0.5 miles (x-axis) vs. median nightly price (y-axis). Bubble size = number of listings. Color = borough. Neighborhoods with fewer than 20 listings excluded.")
st.markdown("This neighborhood scatter plot highlights a lack of within-borough variation, with clear groupings by borough visible throughout. Overall, we see a positive correlation between the average number of nearby stations and median price — but with a clear divide along borough lines, reinforcing that borough-level economic geography drives much of the pattern.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION: CHOROPLETH
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## Choropleth: Median Price by Borough")
st.markdown("""
Stepping back to the borough level gives a clean geographic summary of where prices land across the
city. The following choropleth pairs median price with subway station density — hover over each borough
to see both figures together. Median price is used rather than mean to reduce distortion from extreme
luxury listings at the top of the distribution.
""")

borough_prices = fdf.groupby("neighbourhood_group").agg(
    median_price=("price", "median"),
    mean_price=("price", "mean"),
    listing_count=("price", "count"),
    median_stations=("stations_05mi", "median")
).reset_index()

fig_choro = px.choropleth_mapbox(
    borough_prices,
    geojson=borough_geo,
    locations="neighbourhood_group",
    featureidkey="properties.name",
    color="median_price",
    color_continuous_scale="YlOrRd",
    hover_data={"mean_price": ":$.0f", "listing_count": ":,", "median_stations": True},
    labels={"median_price": "Median Price ($)", "mean_price": "Mean Price ($)",
            "listing_count": "Listings", "median_stations": "Median Stations (0.5mi)"},
)
fig_choro.update_layout(
    mapbox_style="carto-positron",
    mapbox_center={"lat": 40.7128, "lon": -74.0060},
    mapbox_zoom=9.5,
    height=550,
    margin=dict(l=0, r=0, t=0, b=0)
)
st.plotly_chart(fig_choro, use_container_width=True)
st.caption("Figure 11: Choropleth map of median nightly Airbnb price by NYC borough. YlOrRd color scale; darker red = higher median price. Hover for mean price, listing count, and median nearby station count.")
st.markdown("Manhattan's deep red confirms it as the most expensive market by a clear margin. Brooklyn registers a noticeably warmer hue than Queens, the Bronx, and Staten Island — consistent with its premium relative to the outer boroughs. Crucially, Manhattan also carries the highest median subway station count of any borough. This spatial coincidence of high price and high transit density is the central confound the regression section must unpack: are guests paying for transit access, or simply for being in Manhattan?")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION: REGRESSION RESULTS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## Statistical Analysis: OLS Regression")
st.markdown("""
We estimate three OLS regression models to test whether subway proximity predicts Airbnb price,
progressively adding controls following a DAG-based causal framework:

- **Model 1:** Price ~ Stations *(bivariate)*
- **Model 2:** Price ~ Stations + Borough
- **Model 3:** Price ~ Stations + Borough + Room Type + Bedrooms + Beds
""")

# Run regressions on full (unfiltered) data for stable estimates
import statsmodels.api as sm_api

reg_df = df[["price_capped", "stations_05mi", "neighbourhood_group",
             "room_type", "bedrooms", "beds"]].dropna().copy()
reg_df["bedrooms"] = pd.to_numeric(reg_df["bedrooms"], errors="coerce")
reg_df["beds"] = pd.to_numeric(reg_df["beds"], errors="coerce")
reg_df = reg_df.dropna()
reg_df = pd.get_dummies(reg_df, columns=["neighbourhood_group", "room_type"], drop_first=True)
for c in reg_df.columns:
    reg_df[c] = reg_df[c].astype(float)

y = reg_df["price_capped"]
borough_cols = [c for c in reg_df.columns if c.startswith("neighbourhood_group_")]
room_cols = [c for c in reg_df.columns if c.startswith("room_type_")]

X1 = sm_api.add_constant(reg_df[["stations_05mi"]])
m1 = sm_api.OLS(y, X1).fit()

X2 = sm_api.add_constant(reg_df[["stations_05mi"] + borough_cols])
m2 = sm_api.OLS(y, X2).fit()

X3 = sm_api.add_constant(reg_df[["stations_05mi", "bedrooms", "beds"] + borough_cols + room_cols])
m3 = sm_api.OLS(y, X3).fit()

# Display results
r1, r2, r3 = st.columns(3)
r1.metric("Model 1 (bivariate)", f"${m1.params['stations_05mi']:.2f}/station",
          f"R² = {m1.rsquared:.3f}")
r2.metric("Model 2 (+Borough)", f"${m2.params['stations_05mi']:.2f}/station",
          f"R² = {m2.rsquared:.3f}")
r3.metric("Model 3 (Full)", f"${m3.params['stations_05mi']:.2f}/station",
          f"R² = {m3.rsquared:.3f}")

# Coefficient comparison chart
coef_data = pd.DataFrame({
    "Model": ["1: Bivariate", "2: + Borough", "3: Full Controls"],
    "Coefficient": [m1.params["stations_05mi"], m2.params["stations_05mi"], m3.params["stations_05mi"]],
    "CI_low": [m1.conf_int().loc["stations_05mi", 0],
               m2.conf_int().loc["stations_05mi", 0],
               m3.conf_int().loc["stations_05mi", 0]],
    "CI_high": [m1.conf_int().loc["stations_05mi", 1],
                m2.conf_int().loc["stations_05mi", 1],
                m3.conf_int().loc["stations_05mi", 1]],
})

fig_coef = go.Figure()
fig_coef.add_trace(go.Bar(
    y=coef_data["Model"], x=coef_data["Coefficient"],
    orientation="h",
    marker_color=["#2196F3", "#FF9800", "#4CAF50"],
    error_x=dict(
        type="data",
        symmetric=False,
        array=coef_data["CI_high"] - coef_data["Coefficient"],
        arrayminus=coef_data["Coefficient"] - coef_data["CI_low"]
    )
))
fig_coef.add_vline(x=0, line_dash="dash", line_color="black")
fig_coef.update_layout(
    title="Coefficient on 'Stations within 0.5 mi' Across Models",
    xaxis_title="$ per additional station (95% CI)",
    height=350
)
st.plotly_chart(fig_coef, use_container_width=True)

st.markdown(f"""
**Interpretation:** Each additional subway station within 0.5 miles is associated with a
**${m3.params['stations_05mi']:.2f}** change in nightly price, after controlling for borough,
room type, and listing size.
""")

with st.expander("Full Model 3 Regression Table"):
    st.text(m3.summary().as_text())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION: CONCLUSIONS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## Summary & Limitations")

st.markdown("""
### Key Findings

1. **Borough dominates pricing.** Manhattan has the highest prices *and* the densest subway coverage,
   creating a strong confound.
2. **Raw correlation is positive** — more subway stations nearby correlates with higher prices — but
   much of this is driven by borough.
3. **After controlling for borough, room type, and size,** the relationship between subway proximity
   and price is revealed more clearly by the regression.
4. **Room type and listing size** are the strongest individual predictors of price.

### Limitations

- **Observational data** — we cannot claim causation. Subway stations correlate with other
  neighborhood amenities (restaurants, nightlife) that also drive price.
- **Missing variables** — listing quality, photos, specific amenities, and seasonality are not
  captured.
- **Cross-sectional** — a single snapshot in time; prices change seasonally.
- **Distance metric** — straight-line distance differs from walking distance; actual transit
  accessibility depends on service frequency.

### Future Directions

- Incorporate time-series data for seasonal analysis
- Add neighborhood-level controls (median income, walkability, crime)
- Use natural experiments (new station openings) for causal inference
""")

st.markdown("---")
st.markdown("""
### Data Sources

1. [NYC Airbnb Listings (Inside Airbnb / Kaggle)](https://www.kaggle.com/datasets/vrindakallu/new-york-dataset)
2. [MTA Subway Stations (Data.gov)](https://catalog.data.gov/dataset/mta-subway-stations)
3. [NYC Borough Boundaries (GeoJSON)](https://data.insideairbnb.com/united-states/ny/new-york-city/)
""")
