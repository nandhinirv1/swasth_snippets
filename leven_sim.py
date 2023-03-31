from Levenshtein import distance as levenshtein_distance
st1 = "rash"
st2 = "huge rash"

len_st1=len(st1)
len_st2=len(st2)

if len_st1>len_st2:
    max_len=len_st1
else:
    max_len=len_st2

sc=1-(levenshtein_distance(st1,st2))/(max_len)
print(sc)
