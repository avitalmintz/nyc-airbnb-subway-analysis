# NYC Airbnb & Subway Proximity Analysis

An interactive data visualization project investigating whether Airbnb listing prices in New York City are associated with proximity to subway stations.

**[Live App](https://data-visualization-project-bwnhsp4vx5qmga8rdwh2v8.streamlit.app/)** | DATA 22700 - Winter 2026, University of Chicago

## Research Question

*Are Airbnb listing prices in NYC associated with proximity to subway stations, and does this relationship vary across boroughs?*

## Methods

- Merged Airbnb listings data with NYC subway station locations using geospatial distance calculations
- Analyzed price distributions by subway proximity across all five boroughs
- Built regression models controlling for room type, neighborhood, and availability
- Created interactive visualizations with Streamlit and Python plotting libraries

## Key Findings

- Subway proximity has a statistically significant but modest effect on listing prices
- The relationship varies substantially by borough (strongest in Manhattan, weakest in outer boroughs)
- Room type and neighborhood are stronger predictors of price than subway access alone

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Team

Cristian Garcia, Avital Mintz, Vedant Dangayach

## Tech Stack

Python, Streamlit, Pandas, Scikit-learn, Matplotlib/Seaborn
