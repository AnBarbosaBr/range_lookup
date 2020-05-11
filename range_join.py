import numpy as np
import numba as nb
import pandas as pd



def binary_search(scalar,vector):
    n = len(vector)
    start = 0
    end = n-1
    mid = int(np.floor((end-start)/2))
    guard = 0

    while(start < end):
        guard += 1
        if(guard > n):
            break
        if(scalar == vector[mid]):
            return mid
        if(scalar == vector[start]):
            return start

        if(scalar == vector[end]):
            return end

        if(scalar >= vector[mid]):
            start = mid

        if(scalar <= vector[mid]):
            end = mid

        mid = int(start+int((end-start)/2))
    if(scalar == vector[start]):
        return start
    elif(scalar == vector[end]):
        return end
    else:
        return -1

nb_search = nb.jit(binary_search, nopython=True)


lista = np.array(list(range(10**6)))

2000 = 195 / 31 = 6.29
4000 = 219 / 34 = 6.44
10000 = 232 / 38 = 6.1
10000 = 232 / 38 = 6.1
50000 = 319 / 45 = 7
10e6 = 436 / 59 = 7.3
%%timeit
nb_search(10, lista)

lista = [1, 5, 10, 20]
print(nb_search(1, lista))
print(nb_search(5, lista))
print(nb_search(10, lista))
print(nb_search(15, lista))
print(nb_search(20, lista))




def find_first_of_range(scalar,vector):
    n = len(vector)
    start = 0
    end = n-1
    mid = int(np.floor((end-start)/2))
    guard = 0


    first = scalar >= vector[start]
    last = scalar >= vector[end]
    # Ele é maior do que toda a range
    if(last):
        return end
    # Ele é menor que a range
    if(not first):
        return -1

    while(start < end):
        guard += 1
        if(guard > n):
            break
        
        actual = scalar >= vector[mid]
        if(mid > 1):
            prev = scalar >= vector[mid-1]
        else:
            prev = first
        if(prev and not actual):
            return mid-1
        
        if(prev):
            start = mid

        if(not prev):
            end = mid
        mid = int(start+max(int((end-start)/2),1))
    
    return -1



nb_range = nb.jit(find_first_of_range, nopython=True)

np.random.seed(123)
N = 10000
M = 500
a = np.arange(N)
b = np.random.uniform(low = 0, high = N, size = M)

dfA = pd.DataFrame(dict(a=a))
dfB = pd.DataFrame(dict(b=b))
dfB["class"] = np.floor(dfB.b)
dfB = dfB.sort_values("b").drop_duplicates("class")

dfA["key"]=0
dfB["key"]=0


dfA2 = dfA.copy()


%%timeit
dfA2["kb"] = dfA.a.apply((lambda x: nb_range(x, dfB.b.values)))
uniao3 = dfA2.merge(dfB, how="inner", left_on="kb", right_on=dfB.index).drop(["key_x", "kb","key_y"], axis=1)
uniao3.set_index("a", inplace=True)
# 5.35ms, 
# 17.3ms, 19
# 137

%%timeit
dfA2["kb"] = dfA.a.apply((lambda x: find_first_of_range(x, dfB.b.values)))
uniao3 = dfA2.merge(dfB, how="inner", left_on="kb", right_on=dfB.index).drop(["key_x", "kb","key_y"], axis=1)
uniao3.set_index("a", inplace=True)
# 5.48ms
# 43.5

%%timeit
uniao = dfA.merge(dfB, how="inner", on="key").drop("key", axis=1)
uniao = uniao[uniao["a"] > uniao["b"]].groupby("a").last()
# 7.04ms
# 250ms

%%timeit
a1 = dfA.a.values[: , np.newaxis]
b1 = dfB.b.values
i, j = np.where(a1 > b1)

uniao2 = pd.DataFrame(
    np.column_stack([dfA.values[i], dfB.values[j]]),
    columns=dfA.columns.append(dfB.columns)
).drop("key", axis=1).sort_values(["a","b"]).groupby("a").last()

# 5.74ms
# 341 ms