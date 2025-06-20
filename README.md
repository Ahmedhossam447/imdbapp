# 🎬 IMDB Top Movies Analysis Streamlit App

This Streamlit application provides an interactive platform to explore and analyze the IMDB Top 250 movies dataset. Users can delve into various aspects of the dataset, including rating distributions, director insights, genre trends, decade-wise analysis, movie runtimes, and a custom search feature.

## ✨ Features

* **Interactive Filters:** Filter movies by year range, genre, rating, and runtime directly from the sidebar.
* **Rating Distribution:** Visualize the distribution of movie ratings and observe rating trends across different years and decades.
* **Director Analysis:** Discover the most featured directors and those with the highest average ratings.
* **Genre Analysis:** Understand the distribution of different movie genres and their average ratings.
* **Decade Trends:** Explore how movie counts, average ratings, and runtimes have changed over the decades.
* **Runtime Analysis:** Analyze movie runtime distributions, their relationship with ratings, and identify the longest and shortest movies, as well as average runtimes by genre.
* **Custom Search:** Easily search for movies by title, director, or year.
* **Data Download:** Download the filtered dataset for further personal analysis.

## 🚀 How to Run Locally

To run this application on your local machine, follow these steps:

### Prerequisites

* **Python 3.7+**
* **pip** (Python package installer)

### Installation

1.  **Clone the repository (or save the code):**
    If your code is in a Git repository, clone it:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```
    If you just have the `.py` file, save it as `app.py` (or any other name) in a directory.

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required libraries:**
    ```bash
    pip install streamlit pandas plotly numpy
    ```

### Running the Application

After installing the dependencies, run the Streamlit app:

```bash
streamlit run app.py
