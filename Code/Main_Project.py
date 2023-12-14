import requests
import json
import csv


# This class manages city coordinates.
class CityCoordinates:
    def __init__(self):
        self.cities_coordinates = {
            'Vancouver': (49.2827, -123.1207),
            'Toronto': (43.6532, -79.3832),
            'Niagara Falls': (43.0896, -79.0849),
            'Montreal': (45.5017, -73.5673),
            'Whistler': (50.1163, -122.9574),
            'Quebec City': (46.8139, -71.2082),
            'Ottawa': (45.4215, -75.6972),
            'Okanagan Valley': (49.8877, -119.4960),
            'Vancouver Island': (49.6506, -125.4495),
            'Yellowknife': (62.4540, -114.3718),
            'Squamish': (49.7016, -123.1558),
            'Halifax': (44.6488, -63.5752),
            'Calgary': (51.0447, -114.0719),
            'Mont Tremblant': (46.1185, -74.5962),
            'Gananoque': (44.3306, -76.1619),
            'Whitehorse': (60.7212, -135.0568),
            'New York City': (40.7128, -74.0060),
            'Edmonton': (53.5461, -113.4938),
            'Kingston': (44.2312, -76.4860),
            'Kamloops': (50.6745, -120.3273),
            'Buffalo': (42.8864, -78.8784),
            'Fraser Lake': (54.0606, -124.8522),
            'Newfoundland': (47.5615, -52.7126),
            'Markham': (43.8563, -79.3370),
            'Grand Falls-Windsor': (48.9368, -55.6619),
            'Peterborough': (44.3091, -78.3197),
            'North Vancouver': (49.3200, -123.0724),
            'Banff': (51.1784, -115.5708),
            'Trinity': (48.3665, -53.3598),
            'Boston': (42.3601, -71.0589),
            'Lake Louise': (51.4254, -116.1773),
            'Fairbanks': (64.8378, -147.7164),
            'Charlottetown': (46.2382, -63.1311),
            'Port Rexton': (48.3962, -53.3335),
        }

    def get_coordinates(self, city_name):
        return self.cities_coordinates.get(city_name.strip().title())


# This function takes the route passed to it from RouteCalculator and saves it in a JSON and CSV format.
def save_route_data(route, json_file_path, csv_file_path):
    # Saving JSON
    with open(json_file_path, 'w') as file:
        json.dump(route, file, indent=4)

    # Extracting steps and saving to CSV
    coordinates = route['features'][0]['geometry']['coordinates']
    steps = route['features'][0]['properties']['segments'][0]['steps']
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['step', 'start_longitude', 'start_latitude', 'end_longitude', 'end_latitude', 'distance', 'duration'])
        for i, step in enumerate(steps):
            start_index = step['way_points'][0]
            end_index = step['way_points'][1]
            start_lon, start_lat = coordinates[start_index]
            end_lon, end_lat = coordinates[end_index] if end_index < len(coordinates) else coordinates[-1]
            distance = step['distance']
            duration = step['duration']
            writer.writerow([i, start_lon, start_lat, end_lon, end_lat, distance, duration])


# This class uses OpenRouteservice API key retrieves the route taken to travel between the 2 cities and passes it to save_route_data
class RouteCalculator:
    def __init__(self, start_coords, end_coords):
        self.start_coords = start_coords
        self.end_coords = end_coords

    def calculate_route(self):
        url = "https://api.openrouteservice.org/v2/directions/driving-car"
        params = {
            'start': f'{self.start_coords[1]},{self.start_coords[0]}',
            'end': f'{self.end_coords[1]},{self.end_coords[0]}'
        }
        headers = {
            'Authorization': '5b3ce3597851110001cf62489e6e1f5dc3ca4ebc83c9f381c4101f3f'}
        response = requests.get(url, params=params, headers=headers)
        return response.json()


def print_pois(pois):
    header = "{:<90} | {:<10} | {:<15} | {:<10}".format("Name", "Sentiment", "Popularity", "Price Level")
    print(header)
    print("-" * len(header))
    for poi in pois:
        name, sentiment, popularity, price_level = poi['name'], poi['sentiment_rating'], poi['popularity_rating'], \
                                                   poi['price_level']
        formatted_poi = "{:<90} | {:<10} | {:<15} | {:<10}".format(name, sentiment, popularity, price_level)
        print(formatted_poi)


# This class manages the matching of POIs that fall on or near the route and also filtering it according to the user's preference
class POIManager:
    def __init__(self, route_file_path, poi_file_path):
        self.route_file_path = route_file_path
        self.poi_file_path = poi_file_path

    def load_route_from_csv(self):
        route_cords = []
        with open(self.route_file_path, mode='r') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                route_cords.append((float(row['start_latitude']), float(row['start_longitude'])))
        return route_cords

    def load_poi_from_csv(self):
        pois = []
        with open(self.poi_file_path, mode='r') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                lat, lon = float(row['latitude']), float(row['longitude'])
                pois.append({
                    'name': row['name'],
                    'cords': (lat, lon),
                    'sentiment_rating': row.get('sentiment_rating', 'N/A'),
                    'popularity_rating': row.get('popularity_rating', 'N/A'),
                    'price_level': row.get('price_level', 'N/A')
                })
        return pois

    def find_pois_within_route_bounds(self, route_coords, keywords=None):
        # Find min and max coordinates of the route to define a bounding box
        min_lat = min(route_coords, key=lambda x: x[0])[0]
        max_lat = max(route_coords, key=lambda x: x[0])[0]
        min_lon = min(route_coords, key=lambda x: x[1])[1]
        max_lon = max(route_coords, key=lambda x: x[1])[1]

        # Load POIs
        pois = self.load_poi_from_csv()

        pois_within_bounds = []
        for poi in pois:
            poi_lat, poi_lon = poi['cords']
            if min_lat <= poi_lat <= max_lat and min_lon <= poi_lon <= max_lon:
                if keywords is None or any(keyword.lower() in poi['name'].lower() for keyword in keywords):
                    pois_within_bounds.append(poi)

        return pois_within_bounds


# Main Execution
def main():
    city_coord_manager = CityCoordinates()
    start_city = input("Please enter a starting city name: ")
    end_city = input("Please enter an ending city name: ")
    start_coords = city_coord_manager.get_coordinates(start_city)
    end_coords = city_coord_manager.get_coordinates(end_city)

    if not start_coords or not end_coords:
        print("Invalid city name(s). Please check the spelling.")
        return

    route_calculator = RouteCalculator(start_coords, end_coords)
    route = route_calculator.calculate_route()
    save_route_data(route, 'route_data.json', 'route_data.csv')

    # Replace 'route_data.json' (if needed) with the path of that file. Do this for 'route_data.csv' and 'full_cleaned.csv' as well.
    # Do not forget to put r outside the path of the files.

    poi_manager = POIManager('route_data.csv', 'full_cleaned.csv')
    route_coords = poi_manager.load_route_from_csv()
    keywords = [keyword.strip() for keyword in
                input("Please enter the category of establishment you want to visit (Separated by a comma): ").split(',')]
    pois_within_bounds = poi_manager.find_pois_within_route_bounds(route_coords, keywords)
    print_pois(pois_within_bounds)

if __name__ == "__main__":
    main()
