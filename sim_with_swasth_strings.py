from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

# Two lists of sentences
#SEMANTIC SIMILARITY
sentences1 = ['''Referred for rehab - Rt elbow Lateral Epicondylitis
 
 Cozen's test + R
 TREATMENT PLAN
 Physiotherapy - UST / Eccentric strengthening wrist extensors / Manual therapy
 Advised lifestyle modifications and ergonomic adaptations
 
 Ref by Prof MS
 
 PROVISIONAL DIAGNOSIS
 LATERAL EPICONDYLITIS RT ELBOW ''']
sentences2 = ['''Referred for rehab - Rt elbow Lateral Epicondylitis
 
 Cozen's test + R
 TREATMENT PLAN
 Physiotherapy - UST / Eccentric strengthening wrist extensors / Manual therapy
 Advised lifestyle modifications and ergonomic adaptations
 
 Ref by Prof MS
 
 PROVISIONAL DIAGNOSIS
 LATERAL EPICONDYLITIS RT ELBOW ''']

#Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

#Compute cosine-similarities
cosine_scores = util.cos_sim(embeddings1, embeddings2)

#Output the pairs with their score
for i in range(len(sentences1)):
    print(cosine_scores[i][i].item())
    
    
    
#LEVENSTEIN

from Levenshtein import distance as levenshtein_distance
st1 = '''Referred for rehab - Rt elbow Lateral Epicondylitis
 
 Cozen's test + R
 TREATMENT PLAN
 Physiotherapy - UST / Eccentric strengthening wrist extensors / Manual therapy
 Advised lifestyle modifications and ergonomic adaptations
 
 Ref by Prof MS
 
 PROVISIONAL DIAGNOSIS
 LATERAL EPICONDYLITIS RT ELBOW'''
st2 = '''Referred for rehab - Rt elbow Lateral Epicondylitis
 
 Cozen's test + R
 TREATMENT PLAN
 Physiotherapy - UST / Eccentric strengthening wrist extensors / Manual therapy
 Advised lifestyle modifications and ergonomic adaptations
 
 Ref by Prof MS
 
 PROVISIONAL DIAGNOSIS
 LATERAL EPICONDYLITIS RT ELBOW'''

len_st1=len(st1)
len_st2=len(st2)

if len_st1>len_st2:
    max_len=len_st1
else:
    max_len=len_st2

sc=1-(levenshtein_distance(st1,st2))/(max_len)
print(sc)
