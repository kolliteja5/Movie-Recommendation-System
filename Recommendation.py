import numpy as np
#importing the fetcch_movielens dataset
from lightfm.datasets import fetch_movielens
#importing the LightFM package
from lightfm import LightFM

#taking the movielens data set to an object "data":
data = fetch_movielens(min_rating = 4.0)

print(repr(data['train']))
print(repr(data['test']))

#creating a wrap loss fucntion :
model  = LightFM(loss = 'warp')

#training the module
model.fit(data['train'], epochs=30, num_threads = 2)

def sample_recommendation(model, data, user_ids):
    n_users, n_items = data['train'].shape
    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]
        print("User %s" % user_id)
        print("     Known positives:")
        for x in known_positives[:3]:
            print("        %s" % x)
        print("     Recommended:")
        for x in top_items[:3]:
            print("        %s" % x)
            
sample_recommendation(model, data, [3, 25, 451])            
        
            
            
            
            
            