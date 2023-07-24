import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import math
import numpy as np
from shapely.geometry import Point
import random
import datetime
import geopandas as gpd
import multiprocessing
import gtfs_kit as gk
import geopy.distance
import os

def osmnx_routing_graph(city_center,radius):
    G = ox.graph.graph_from_point(city_center,dist=radius,
                            dist_type = "bbox",
                            network_type='drive',
                            simplify=True,
                            truncate_by_edge=True,
                            retain_all=False)
    # G = ox.graph_from_bbox(north=35.4333303354069, south=34.98355788426319, east=-84.9464023036812, west=-85.408215,
    #                         network_type='drive',
    #                         simplify=True,
    #                         truncate_by_edge=True,
    #                         retain_all=False)
    G = ox.utils_graph.get_largest_component(G, strongly=True)

    # add edge speeds
    G = ox.speed.add_edge_speeds(G, fallback=40.2, precision=6)

    # add edge travel time
    G = ox.speed.add_edge_travel_times(G, precision=6)
    for n1, n2, k in G.edges(keys=True):
        G[n1][n2][k]['travel_time'] = math.ceil(G[n1][n2][k]['travel_time'])


    nodes, edges = ox.utils_graph.graph_to_gdfs(G)

    # format nodes
    nodes['osmid'] = nodes.index
    nodes.index = range(len(nodes))
    nodes['node_id'] = nodes.index
    nodes['lon'] = nodes['x']
    nodes['lat'] = nodes['y']
    nodes = nodes[['node_id', 'osmid', 'lat', 'lon']]
    nodes['node_id'] = nodes['node_id'].astype(int)
    nodes['osmid'] = nodes['osmid'].astype(int)
    nodes['lat'] = nodes['lat'].astype(float)
    nodes['lon'] = nodes['lon'].astype(float)
    nodes['node_id'] = nodes['node_id'].apply(lambda x: x + 1)

    # format edges
    edges = edges.reset_index()
    edges['source_osmid'] = edges['u']
    edges['target_osmid'] = edges['v']
    edges['source_node'] = edges['source_osmid'].apply(lambda x: nodes.loc[nodes['osmid']==x, 'node_id'].values[0])
    edges['target_node'] = edges['target_osmid'].apply(lambda x: nodes.loc[nodes['osmid']==x, 'node_id'].values[0])
    edges = edges.sort_values(by=['travel_time'])
    edges = edges.drop_duplicates(subset=['source_node', 'target_node'])
    edges = edges[['source_osmid', 'target_osmid', 'source_node', 'target_node', 'travel_time']]
    edges['source_osmid'] = edges['source_osmid'].astype(int)
    edges['target_osmid'] = edges['target_osmid'].astype(int)
    edges['source_node'] = edges['source_node'].astype(int)
    edges['target_node'] = edges['target_node'].astype(int)
    edges['travel_time'] = edges['travel_time'].astype(int)
    # format edge types
    print(f"Number of nodes: {len(nodes)}, number of edges: {len(edges)}")
    return G, nodes, edges

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def read_matrix(filename,num_lines):
    delimiter = ','
    dtype = np.uint16
    def iter_func():
        with open(filename, 'r') as infile:
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, num_lines))
    return data

def generateMap(G,nodes,edges,main_directory):
    OUTPUT_DIR = main_directory+"map/"+radius_in_km
    create_directory(OUTPUT_DIR)

    node_to_osmid_map = {}
    for _,node in nodes.iterrows():
        node_to_osmid_map[int(node.node_id)] = node.osmid

    osmid_to_node_map = {}
    for _,node in nodes.iterrows():
        osmid_to_node_map[node.osmid] = int(node.node_id)

    with open(OUTPUT_DIR+"pred.csv", 'a+') as pred_file:
        with open(OUTPUT_DIR+"times.csv", 'a+') as times_file:
            for origin in range(1,len(nodes)+1):
                travel_times = []
                predecessors = []
                origin_osmid = node_to_osmid_map[origin]
                pred,travel_time=nx.dijkstra_predecessor_and_distance(G, origin_osmid,weight='travel_time')
                for destination in range(1,len(nodes)+1):
                    destination_osmid = node_to_osmid_map[destination]
                    travel_times.append(int(travel_time[destination_osmid]))
                    if destination == origin:
                        predecessor = 0
                    else:
                        predecessor = osmid_to_node_map[pred[destination_osmid][0]]
                    predecessors.append(predecessor)
                pred_file.write(",".join([str(i) for i in predecessors])+"\n")
                times_file.write(",".join([str(i) for i in travel_times])+"\n")

    times = read_matrix(OUTPUT_DIR+'times.csv',len(nodes))
    with open(OUTPUT_DIR +"times.npy", 'wb') as f:
        np.save(f, times)

    pred = read_matrix(OUTPUT_DIR+'pred.csv',len(nodes))
    with open(OUTPUT_DIR +"pred.npy", 'wb') as f:
        np.save(f, pred)
    # predecessors = np.genfromtxt(OUTPUT_DIR+'pred.csv', delimiter=',', dtype=np.int16)
    # with open(OUTPUT_DIR+"distance.csv", 'a+') as dist_file:
    #     for origin in range(1,len(nodes)+1):
    #         distances = []
    #         for destination in range(1,len(nodes)+1):
    #             distance = 0
    #             if destination != origin:
    #                 current_target = destination
    #                 current_target_osmid = nodes.loc[nodes['node_id']==destination,'osmid'].iloc[0]
    #                 while True:
    #                     next_target = predecessors[origin-1,current_target-1]
    #                     next_target_osmid = nodes.loc[nodes['node_id']==next_target,'osmid'].iloc[0]
    #                     distance += G[next_target_osmid][current_target_osmid][0]['length']
    #                     if origin == next_target:
    #                         break
    #                     current_target = next_target
    #                     current_target_osmid = next_target_osmid
    #             distances.append(distance)
    #         dist_file.write(",".join([str(i) for i in distances])+"\n")

    nodes = nodes[['node_id', 'lat', 'lon']]
    nodes.to_csv(OUTPUT_DIR+'nodes.csv', header=False, index=False)

    edges['source_node'] = edges['source_node'].apply(lambda x: x + 1)
    edges['target_node'] = edges['target_node'].apply(lambda x: x + 1)
    edges['travel_time'] = edges['travel_time'].apply(lambda x: math.ceil(x))
    edges = edges[['source_node', 'target_node', 'travel_time']]
    edges.to_csv(OUTPUT_DIR+'edges.csv', header=False, index=False)

def generateVehicles(nodes,vehicle_num,vehicle_capacity,main_directory):
    nodes = nodes[['node_id', 'lat', 'lon']]
    vehicles = pd.DataFrame()
    # randomly generate starting points 
    start_node = random.sample(nodes.index.to_list(), vehicle_num)
    print(nodes.iloc[start_node[0]])
    for n in start_node:
        vehicles = vehicles.append(nodes.iloc[n], ignore_index=True)

    vehicles.columns = ['node', 'lat', 'lon']
    vehicles['id'] = list(range(1, vehicle_num+1))
    vehicles['start_time'] = datetime.time(0,0,0)
    vehicles['capacity'] = vehicle_capacity

    # formatting
    vehicles['node'] = vehicles['node'].astype('int')
    vehicles = vehicles[['id', 'node', 'lat', 'lon', 'start_time', 'capacity']]
    vehicle_directory = main_directory + "vehicles/"+radius_in_km
    create_directory(vehicle_directory)
    vehicles.to_csv(vehicle_directory+'vehicles.csv', index = False)


def extract_requests_from_lodes_data(cutoff,center,block_file_path,lodes_file_path,main_directory):
    request_directory = main_directory+"requests/"+radius_in_km
    create_directory(request_directory)
    ma_blocks = gpd.read_file(block_file_path)
    ma_blocks.GEOID20 = ma_blocks.GEOID20.astype(int)

    ma_lodes = pd.read_csv(lodes_file_path).rename(columns = {'S000':'total_jobs'})
    # ma_lodes.w_geocode = ma_lodes.w_geocode.astype(str)
    # ma_lodes.h_geocode = ma_lodes.h_geocode.astype(str)
    ma_lodes = ma_lodes.groupby(['h_geocode', 'w_geocode']).agg(total_jobs=('total_jobs', sum)).reset_index().merge(ma_blocks[['GEOID20', 'geometry']], left_on='h_geocode', right_on='GEOID20').rename({'geometry':'home_geom'}, axis=1).drop('GEOID20', axis=1).merge(ma_blocks[['GEOID20', 'geometry']], left_on='w_geocode', right_on='GEOID20').rename({'geometry':'work_geom'}, axis=1).drop('GEOID20', axis=1).sort_values('total_jobs', ascending=False).reset_index(drop=True)

    with open(request_directory+"temp_requests.csv", 'a+') as req_file:
        req_file.write("id,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude\n")
        id = 0
        for _,row in ma_lodes.iterrows():
            total_jobs = row.total_jobs
            while total_jobs>0:
                minx, miny, maxx, maxy = row.home_geom.bounds
                ori_lon,ori_lat = np.random.uniform(minx, maxx), np.random.uniform(miny, maxy)
                coords = (ori_lat,ori_lon)
                distance_to_o = geopy.distance.geodesic(center, coords).km
                minx, miny, maxx, maxy = row.work_geom.bounds
                des_lon,des_lat = np.random.uniform(minx, maxx), np.random.uniform(miny, maxy)
                coords = (des_lat,des_lon)
                distance_to_d = geopy.distance.geodesic(center, coords).km
                if distance_to_o <= cutoff and distance_to_d <= cutoff:
                    req_file.write("{0},{1},{2},{3},{4}\n".format(id,ori_lon,ori_lat,des_lon,des_lat))
                    id+=1
                total_jobs-=1
        print("Saved temp requests")

def generate_requests(G,nodes,main_directory):
    osmid_to_node_map = {}
    for _,node in nodes.iterrows():
        osmid_to_node_map[node.osmid] = int(node.node_id)
    request_directory = main_directory+"requests/"+radius_in_km
    create_directory(request_directory)
    requests = pd.read_csv(request_directory+"temp_requests.csv")
    no_of_requests = requests.shape[0]
    time_gap = 7200/no_of_requests
    id = 1
    starting_time = datetime.datetime(year=2023,month=3,day=28,hour=6,minute=0,second=0)
    with open(request_directory+"requests.csv", 'a+') as req_file:
        req_file.write("id,tpep_pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,origin,dest\n")
        for _,row in requests.iterrows():
            new_time = starting_time + datetime.timedelta(seconds=int(time_gap*id))
            time_str = new_time.strftime("%Y-%m-%d %H:%M:%S")
            ori_lon,ori_lat = row.pickup_longitude, row.pickup_latitude
            des_lon,des_lat = row.dropoff_longitude, row.dropoff_latitude
            origin,dest = get_origin_destination(G,osmid_to_node_map,ori_lat,ori_lon,des_lat,des_lon)
            req_file.write("{0},{1},{2},{3},{4},{5},{6},{7}\n".format(id,time_str,ori_lon,ori_lat,des_lon,des_lat,int(origin),int(dest)))
            id += 1
        starting_time = datetime.datetime(year=2023,month=3,day=28,hour=16,minute=0,second=0)
        for _,row in requests.iterrows():
            new_time = starting_time + datetime.timedelta(seconds=int(time_gap*(id-no_of_requests)))
            time_str = new_time.strftime("%Y-%m-%d %H:%M:%S")
            ori_lon,ori_lat = row.pickup_longitude, row.pickup_latitude
            des_lon,des_lat = row.dropoff_longitude, row.dropoff_latitude
            origin,dest = get_origin_destination(G,osmid_to_node_map,ori_lat,ori_lon,des_lat,des_lon)
            req_file.write("{0},{1},{2},{3},{4},{5},{6},{7}\n".format(id,time_str,des_lon,des_lat,ori_lon,ori_lat,int(dest),int(origin)))
            id += 1

def get_origin_destination(G,osmid_to_node_map,ori_lat,ori_lon,des_lat,des_lon):
    ori = ox.distance.nearest_nodes(G, ori_lon, ori_lat)
    des = ox.distance.nearest_nodes(G, des_lon, des_lat)
    origin = None
    destination = None
    if type(ori) == list:
        origin = ori[0]
    else:
        origin = ori
    if type(des) == list:    
        destination = des[0]
    else:
        destination = des
    origin = osmid_to_node_map[int(origin)]
    destination = osmid_to_node_map[int(destination)]
    return origin, destination

def update_bus_stops(G,nodes,main_bus_directory,input_directory):
    stops = pd.read_csv(input_directory+"stops.txt")
    osmid_to_node_map = {}
    for _,node in nodes.iterrows():
        osmid_to_node_map[int(node.osmid)] = node.node_id
    bus_directory = main_bus_directory + radius_in_km
    create_directory(bus_directory)
    with open(bus_directory+"stop_map.csv", 'a+') as stop_file:
        stop_file.write("stop_id,node_id\n")
        for _,row in stops.iterrows():
            if not (np.isnan(row.stop_lon) or np.isnan(row.stop_lat)):
                stop_node = ox.distance.nearest_nodes(G, row.stop_lon, row.stop_lat)
                if type(stop_node) == list:
                    stop_node = stop_node[0]
                stop_node = int(osmid_to_node_map[int(stop_node)])
                stop_file.write("{0},{1}\n".format(row.stop_id,stop_node))

def generate_eligible_lines(G,nodes,cut_off,main_bus_directory,input_directory):
    bus_directory = main_bus_directory + radius_in_km
    create_directory(bus_directory)
    feed = gk.read_feed(input_directory, dist_units='km')
    stop_map = pd.read_csv(bus_directory+"stop_map.csv")
    stop_map_dic = {}
    for _,row in stop_map.iterrows():
        stop_map_dic[row.stop_id] = int(row.node_id)

    selected_day = None
    for day in feed.get_first_week():
        year = int(day[:4])
        month = int(day[4:6])
        date = int(day[6:])
        date_time_obj = datetime.datetime(year=year,month=month,day=date)
        if date_time_obj.weekday() == 0:
            selected_day = day
            break
    # selected_day='20230314'
    # selected_day="20221128"
    busslines = {}
    for _,row in feed.routes.iterrows():
        route_id = row.route_id
        timetable = feed.build_route_timetable(route_id, [selected_day])
        if timetable.shape[0] > 0:
            busslines[route_id] = {}
            for direction_id in timetable.direction_id.unique():
                stops_in_direction = timetable[timetable.direction_id == direction_id].stop_id.unique()
                if stops_in_direction.shape[0] > 0:
                    busslines[route_id][direction_id] = [stop_map_dic[stop] for stop in stops_in_direction]
            print(route_id)

    node_id_to_osmid = {}
    for _,node in nodes.iterrows():
        node_id_to_osmid[int(node.node_id)] = node.osmid

    osmid_to_nodeid = {}
    for _,node in nodes.iterrows():
        osmid_to_nodeid[node.osmid] = int(node.node_id)

    with open(bus_directory+"eligible_lines.csv","a+") as output_file:
        output_file.write("node,line,direction,stop\n")
        for i in range(1,nodes.shape[0]+1):
            res = nx.single_source_dijkstra(G,node_id_to_osmid[i],cutoff=cut_off,weight='length')[0]
            for bus_line_name in busslines:
                for direction in busslines[bus_line_name]:
                    stops = busslines[bus_line_name][direction]
                    closest_stop = stops[0]
                    closest_distance = cut_off+1
                    for stop in stops:
                        stop_osmid = node_id_to_osmid[stop]
                        if stop_osmid in res:
                            dist_to_stop = res[stop_osmid]
                            if dist_to_stop < closest_distance:
                                closest_distance = dist_to_stop
                                closest_stop = stop

                    if closest_distance <= cut_off:
                        output_file.write("{0},{1},{2},{3}\n".format(i,bus_line_name,direction,closest_stop))


def generateRequests1(G,nodes,filename,output_file_name):

    DROPOFF_LAT = 'work_loc_lat'
    DROPOFF_LONG = 'work_loc_lon'
    PICKUP_LAT = 'home_loc_lat'
    PICKUP_LONG = 'home_loc_lon'
    GO_TIME = 'go_time'
    RETURN_TIME = 'return_time'
    TRIP_DISTANCE = 'trip_distance'
    ORIGIN = 'origin'
    DEST = 'dest'

    data = pd.read_csv(filename,usecols=[GO_TIME,RETURN_TIME, PICKUP_LONG, PICKUP_LAT, DROPOFF_LONG, DROPOFF_LAT])
    count = data.shape[0]

    print("Trips in selected time interval: {0}".format(count))
    
    with open("chattanooga/requests"+output_file_name+".csv", 'a') as f:
        f.write('id,tpep_pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,origin,dest\n')
    selected_trips = 0
    num_processed = 0
    trip_id = 1
    node_id_to_osmid = {}
    for _,node in nodes.iterrows():
        node_id_to_osmid[int(node.node_id)] = node.osmid

    osmid_to_nodeid = {}
    for _,node in nodes.iterrows():
        osmid_to_nodeid[node.osmid] = int(node.node_id)

    for index, row in data.iterrows():
            print(num_processed, end='\r')
            num_processed+=1
            go_time = row[GO_TIME]
            return_time = row[RETURN_TIME]
            ori = ox.distance.nearest_nodes(G, row[PICKUP_LONG],row[PICKUP_LAT])
            des = ox.distance.nearest_nodes(G, row[DROPOFF_LONG], row[DROPOFF_LAT])
            origin = None
            destination = None
            if type(ori) == list:
                origin = ori[0]
            else:
                origin = ori
            if type(des) == list:    
                destination = des[0]
            else:
                destination = des
                
            origin = osmid_to_nodeid[origin]
            destination = osmid_to_nodeid[destination]
            if origin != destination:
                selected_trips+=1
                with open("chattanooga/requests"+output_file_name+".csv", 'a') as f:
                    f.write('{0},{1},{2},{3},{4},{5},{6},{7}\n'.format(trip_id,go_time,row[PICKUP_LONG],row[PICKUP_LAT],row[DROPOFF_LONG], row[DROPOFF_LAT],origin,destination))
                    trip_id+=1
                    f.write('{0},{1},{2},{3},{4},{5},{6},{7}\n'.format(trip_id,return_time,row[DROPOFF_LONG], row[DROPOFF_LAT],row[PICKUP_LONG],row[PICKUP_LAT],destination,origin))
                    trip_id+=1

    print("Trips after filtering: {0}".format(selected_trips))

main_directory = "output/"
radius = 12000 #radius in meters
radius_in_km = "{0}km/".format(int(radius//1000))
city_center = (42.36018755809968, -71.05892310513804)
G, nodes, edges = osmnx_routing_graph(city_center,radius)
print(len(nodes),len(edges))
generateMap(G,nodes,edges,main_directory)
generateVehicles(nodes,10000,4,main_directory)
bus_input_directory = main_directory+ "bus/"
update_bus_stops(G,nodes,bus_input_directory,bus_input_directory)
generate_eligible_lines(G,nodes,4000,bus_input_directory,bus_input_directory)

block_file_path = main_directory+"tl_2020_25_tabblock20/tl_2020_25_tabblock20.shp"
lodes_file_path = main_directory+"ma_od_main_JT00_2019.csv"
extract_requests_from_lodes_data(radius/1000,city_center,block_file_path,lodes_file_path,main_directory)
generate_requests(G,nodes,main_directory)

# atlanta downtown 33.75508,-84.38869 #15km
# chicago downtown 41.881978735974656, -87.6301110441199 #16km
# boston downtown 42.36018755809968, -71.05892310513804 #12km
# houston downtown 29.74235,-95.37086 #12km
# la downtown 34.04529,-118.24996 #12km #11771
# generateRequests1(G,nodes,'../taxi-public-transit-integration/generate_data/lodes_combinations.csv',"requests_23_6_23_new")
