from KNN_ALGO import knn, euclidean_distance

def recommend_movies(movie_query, k_recommendations):
    raw_movies_data = []
    with open('patents_recommendation_data.csv', 'r') as md:
#    with open('movies_recommendation_data.csv', 'r') as md:
        # Discard the first line (headings)
        next(md)

        # Read the data into memory
        for line in md.readlines():
            data_row = line.strip().split(',')
            raw_movies_data.append(data_row)

    # Prepare the data for use in the knn algorithm by picking
    # the relevant columns and converting the numeric columns
    # to numbers since they were read in as strings
    movies_recommendation_data = []
    for row in raw_movies_data:
        data_row = list(map(float, row[1:]))
        movies_recommendation_data.append(data_row)
        
    print(movies_recommendation_data)

    # Use the KNN algorithm to get the 5 movies that are most
    # similar to The Post.
    recommendation_indices, _ = knn(
        movies_recommendation_data, movie_query, k=k_recommendations,
        distance_fn=euclidean_distance, choice_fn=lambda x: None
    )

    movie_recommendations = []
    for _, index in recommendation_indices:
        movie_recommendations.append(raw_movies_data[index])

    return movie_recommendations

if __name__ == '__main__':
    target_patent_kw_freq = [460.0, 335.0, 306.0, 165.0, 103.0, 101.0, 95.0, 77.0, 72.0, 70.0, 54.0, 51.0, 48.0, 45.0, 42.0] # feature vector for The Post
#    target_patent_kws=[traffic, signal, vehicle, plurality, data, period, state, communication, signals, location, modification, travel, information, network, electronic]
    target_patent = ['10115305', 'Optimizing autonomous car"\'"s driving time and user experience using traffic signal information']


    recommended_movies = recommend_movies(movie_query=target_patent_kw_freq, k_recommendations=5)

    # Print recommended movie titles
    print('Target Patent Number:', target_patent[0])
    for recommendation in recommended_movies:
        print(recommendation[0])
