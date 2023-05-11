Steps for generating data for the simulator
- Download the LODES data from https://lehd.ces.census.gov/data/
    - Select the state from the `LEHD Origin-Destination Employment Statistics (LODES)` section
    - Click on view files
    - Download the file of the following format `{STATE}_od_main_JT00_{YEAR}.csv.gz`
    - Extract the CSV file
- Download the Census Block reference file from from https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2010&layergroup=Blocks
    - Select the state and click download
    - Alternatively, visit https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.2010.html#list-tab-790442341
        - Select year -> Click on 'FTP Archive by State' -> Select State -> Select the first folder (folder for the whole state) -> Download the file with following format `tl_2010_{State Index}_tabblock10.zip`
