import math
import numpy as np

chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
alpha = 0.5

def getFreq (o_class, start_num, end_num):
    data = {}
    for i in range(start_num, end_num):
        fname = o_class + str(i) + '.txt';
        file_ = open(fname, 'r')
        f_data = file_.read()
        for char in chars:
            if data.get(char) != None:
                data[char] += f_data.count(char)
            else:
                data[char] = f_data.count(char)
    return data

print(len(chars))

p_e = 10.5/(30+3*0.5)
p_s = 10.5/(30+3*0.5)
p_j = 10.5/(30+3*0.5)
print("Priors = ", p_e, p_s, p_j)
p_e = math.log(p_e)
p_s = math.log(p_s)
p_j = math.log(p_j)
print("Priors log = ", p_e, p_s, p_j)

dict_e = getFreq('e', 0, 10)
print(dict_e)
total_e = sum(dict_e.values())

dict_s = getFreq('s', 0, 10)
total_s = sum(dict_s.values())

dict_j = getFreq('j', 0, 10)
total_j = sum(dict_j.values())

#for e
print("likelihood e = ")
for char in chars:
    print(char, (dict_e[char]+alpha)/(total_e+len(chars)*alpha))
    dict_e[char] = (dict_e[char]+alpha)/(total_e+len(chars)*alpha)
#for s
print("likelihood s = ")
for char in chars:
    print(char, (dict_s[char]+alpha)/(total_s+len(chars)*alpha))
    dict_s[char] = (dict_s[char]+alpha)/(total_s+len(chars)*alpha)
#for j
print("likelihood j = ")
for char in chars:
    print(char, (dict_j[char]+alpha)/(total_j+len(chars)*alpha))
    dict_j[char] = (dict_j[char]+alpha)/(total_j+len(chars)*alpha)

for o_class in ['e', 's', 'j']:
    for i in range(10, 20):
        dict_pred = getFreq(o_class, i, i+1)
        print(o_class+ str(i), " bag of words = ")
        for char in chars:
            print(char, (dict_pred[char]))

        p_x_given_y_e = 0
        p_x_given_y_s = 0
        p_x_given_y_j = 0
        for char in chars:
            p_x_given_y_e += dict_pred[char]*math.log(dict_e[char])
            p_x_given_y_s += dict_pred[char]*math.log(dict_s[char])
            p_x_given_y_j += dict_pred[char]*math.log(dict_j[char])

        print(p_x_given_y_e)
        print(p_x_given_y_s)
        print(p_x_given_y_j)
        p_e_given_x = p_x_given_y_e + p_e
        p_s_given_x = p_x_given_y_s + p_s
        p_j_given_x = p_x_given_y_j + p_j
        print(p_e_given_x)
        print(p_s_given_x)
        print(p_j_given_x)
        class_ = np.argmax([p_e_given_x, p_s_given_x, p_j_given_x])
        print("Prediction = ", class_)
