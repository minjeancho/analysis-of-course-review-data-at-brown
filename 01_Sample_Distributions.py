import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

def json_to_dict_array(file_name):
    json     = pd.read_json(file_name)
    js_array = np.array(json)
    # In js_array, multiple records in a row.
    # Thus, reshaping required.
    np_array = js_array.reshape(-1, 1)
    # In np_array, one record per row, but some row have [None]     
    # Thus, removing empty [None] rows
    indices_of_rows_with_records = [ ]
    for i in range(np_array.shape[0]):
        if np_array[i] != None:
            indices_of_rows_with_records.append(i)
    record_array = np_array[indices_of_rows_with_records]
    # The records with no 'test-value' field will be deleted.
    indices_of_rows_with_test_value_field = [ ]    
    for i in range(record_array.shape[0]):
        dict_i  = record_array[i][0]
        if 'test-value' in dict_i.keys():
            indices_of_rows_with_test_value_field.append(i)
    dict_array = record_array[indices_of_rows_with_test_value_field]
    print(file_name, ": no. of valid records:", dict_array.shape[0])
    return dict_array

#dict_array = json_to_dict_array("CSCI.json")

def avg_score_from_value_list(value_list):
    string_list = [value for value in value_list if value != ""]
    number_list = [int(string) for string in string_list if string.isdigit() == True]
    # Invialid entry checked.
    # Only INTEGER entries will be used.
    # A few entries in min_time and max_time contains "12-15" or "12--15" or "a lot"
    # Such entries will be removed before calculating the averages.    
    if len(number_list) != 0: avg_score = np.mean(number_list)
    else:                     avg_score = "NaN"
    return avg_score

#value_list = ["", "1", "12--15", "12-15", "A LOT"]
#print(avg_score_from_value_list(value_list))

def avg_score_from_grade_list(grade_list):
    string_list = [value for value in grade_list if value != ""]
    number_list = [ ]
    for string in string_list:
        if   string == "A": number_list.append(4)
        elif string == "B": number_list.append(3)
        elif string == "C": number_list.append(2)
    # Only {A, B, C} will be used. 
    if len(number_list) != 0: avg_score = np.mean(number_list)
    else:                     avg_score = "NaN"
    return avg_score

#grade_list = ["A", "", "S", "B"]
#print(avg_score_from_grade_list(grade_list))
#file_name = 'CSCI.json'

def dict_array_to_score_array(dict_array):    
    # column  0: 'readings'
    # column  1: 'class-materials'
    # column  2: 'difficult'
    # column  3: 'learned'
    # column  4: 'loved'
    # column  5: 'grading-speed'
    # column  6: 'grading-fairness'
    # column  7: 'effective'
    # column  8: 'efficient'
    # column  9: 'encouraged'
    # column 10: 'passionate'
    # column 11: 'receptive'
    # column 12: 'availableFeedback'
    # column 13: 'grade'
    # column 14: 'minhours'
    # column 15: 'maxhours'
    # column 16: 'class_size'
    score_array = np.empty((dict_array.shape[0], len(features)), dtype=float)
    n_columns = len(features)
    for i in range(dict_array.shape[0]):
        value_dict_i = dict_array[i][0]['test-value']
        for j in range(n_columns - 1):
            key_j = features[j]
            if key_j in value_dict_i.keys():
                if key_j != 'grade':
                    avg_score_j = avg_score_from_value_list(value_dict_i[key_j])
                elif key_j == 'grade':
                    avg_score_j = avg_score_from_grade_list(value_dict_i[key_j])
            else:
                avg_score_j = "NaN"
            score_array[i][j] = avg_score_j
    for i in range(score_array.shape[0]):
        frosh_count = int(dict_array[i][0]['frosh'])
        soph_count  = int(dict_array[i][0]['soph'])
        jun_count   = int(dict_array[i][0]['jun'])
        sen_count   = int(dict_array[i][0]['sen'])
        grad_count  = int(dict_array[i][0]['grad'])
        class_size  = frosh_count + soph_count + jun_count + sen_count + grad_count
        score_array[i][-1] = class_size
    return score_array

#score_array, column_titles = dict_array_to_score_array(dict_array)
#print()
#print("Column titles:/n", column_titles)
#print()

def score_array_to_dataframe(score_array):
    df_dict = dict()
    for i in range(len(features)):
        column_title = features[i]
        value_vector = score_array[:, i]
        df_dict[column_title] = value_vector
    df = pd.DataFrame(df_dict)
    return df
    """
    # Following is equivalent to above.
    df = pd.DataFrame({'readings':          score_array[:, 0],
                       'class-materials':   score_array[:, 1],
                       'difficult':         score_array[:, 2],
                       'learned':           score_array[:, 3],
                       'loved':             score_array[:, 4],
                       'grading-speed':     score_array[:, 5],
                       'grading-fairness':  score_array[:, 6],
                       'effective':         score_array[:, 7],
                       'efficient':         score_array[:, 8],
                       'encouraged':        score_array[:, 9],
                       'passionate':        score_array[:,10],
                       'receptive':         score_array[:,11],
                       'availableFeedback': score_array[:,12],
                       'grade':             score_array[:,13],
                       'minhours':          score_array[:,14],
                       'maxhours':          score_array[:,15],
                       'class_size':        score_array[:,16]})
    return df
    """
#===============================
# Weighted Percentile Calculator
#===============================

def weighted_percentile(array, percentile, sample_weights):
    # percentile range: 0-100
    repeated_samples = np.repeat(array, sample_weights)
    percentile = np.percentile(repeated_samples, percentile)
    return percentile

########################## MAIN ###############################

features = ['readings','class-materials','difficult','learned',
            'loved','grading-speed','grading-fairness',
            'effective','efficient','encouraged','passionate',
            'receptive','availableFeedback',
            'grade','minhours','maxhours','class_size']

areas = ["AFRI", "ANTH", "APMA", "ARAB", "ARCH",
         "BIOL", "CHEM", "CLAS", "CLPS", "COLT",
         "CSCI", "DEVL", "EAST", "ECON", "EDUC",
         "EGYT", "ENGL", "ENGN", "ENVS", "ETHN",
         "FREN", "GEOL", "GNSS", "GRMN", "HIAA",
         "HIST", "INTL", "ITAL", "JUDS", "KREA",
         "LITR", "MATH", "MCM",  "MDVL", "MES",
         "MUSC", "NEUR", "PHIL", "PHP",  "PHYS",
         "PLCY", "POBS", "POLS", "RELS", "REMS",
         "SLAV", "SOC",  "SPAN", "TAPS", "URBN",
         "VISA"]

BROWN_score_array = np.empty((0, len(features)), dtype=float)
for area in areas:
    file_name = area + ".json"
    dict_array  = json_to_dict_array(file_name)
    score_array = dict_array_to_score_array(dict_array)
    BROWN_score_array = np.vstack((BROWN_score_array, score_array))
print()
print("BRWN_score_array generated")
print()
print("df_BROWN before removing rows with missing values")
df_BROWN = score_array_to_dataframe(BROWN_score_array)
print(df_BROWN)
print()
print("df_BROWN after removing rows with missing values")
df_BROWN = df_BROWN.dropna(axis=0)
print(df_BROWN)


x_name = "minhours"
y_name = "grade"
z_name = "difficult"
weight_name = "class_size"

x = df_BROWN[x_name]
y = df_BROWN[y_name]
z = df_BROWN[z_name]
wt = df_BROWN[weight_name] 


x_wQ1 = weighted_percentile(x, 25, wt)
x_wQ2 = weighted_percentile(x, 50, wt)
x_wQ3 = weighted_percentile(x, 75, wt)
x_wIQR = x_wQ3 - x_wQ1
x_outlier_lower = x_wQ1 - (1.5 * x_wIQR)
x_outlier_upper = x_wQ3 + (1.5 * x_wIQR)
print("===============================")
print(x_name)
print("-------------------------------")
print("Q1     :", x_wQ1) # 3.1842
print("Median :", x_wQ2) # 4.1404
print("Q3     :", x_wQ3) # 5.5652
print("-------------------------------")
print("Outlier lower:", x_outlier_lower) # -0.3872
print("Outlier upper:", x_outlier_upper) # +9.1367
print("===============================")
plt.figure(figsize=(5,5))
plt.hist(x, bins=70, density=1, rwidth=0.8, weights=wt)
plt.axvline(x_outlier_lower, color='k', linewidth=1)
#plt.axvline(x_wQ2, color='b', linestyle='dashed', linewidth=1)
plt.axvline(x_outlier_upper, color='k', linewidth=1)
#plt.xlim(0)
plt.xlabel(x_name, size=12)
plt.ylabel("Density", size=10)
#plt.title("Weighted Histogram: minhours")
plt.tight_layout()
plt.savefig("FIG_Histogram_minhours_raw.png")
plt.show()

y_wQ1 = weighted_percentile(y, 25, wt)
y_wQ2 = weighted_percentile(y, 50, wt)
y_wQ3 = weighted_percentile(y, 75, wt)
y_wIQR = y_wQ3 - y_wQ1
y_outlier_lower = y_wQ1 - (1.5 * y_wIQR)
y_outlier_upper = y_wQ3 + (1.5 * y_wIQR)
print("===============================")
print(y_name)
print("-------------------------------")
print("Q1     :", y_wQ1) # 3.5385
print("Median :", y_wQ2) # 3.7127
print("Q3     :", y_wQ3) # 3.8537
print("-------------------------------")
print("Outlier lower:", y_outlier_lower) # 3.0657 
print("Outlier upper:", y_outlier_upper) # 4.3264
print("===============================")

plt.figure(figsize=(5,5))
plt.hist(y, bins=70, density=1, rwidth=0.8, weights=wt)
plt.axvline(y_outlier_lower, color='k', linewidth=1)
#plt.axvline(y_wQ2, color='b', linestyle='dashed', linewidth=1)
plt.axvline(y_outlier_upper, color='k', linewidth=1)
#plt.xlim(0)
plt.xlabel(y_name, size=12)
plt.ylabel("Density", size=10)
#plt.title("Weighted Histogram: grade")
plt.tight_layout()
plt.savefig("FIG_Histogram_grade_raw.png")
plt.show()

z_wQ1 = weighted_percentile(z, 25, wt)
z_wQ2 = weighted_percentile(z, 50, wt)
z_wQ3 = weighted_percentile(z, 75, wt)
z_wIQR = z_wQ3 - z_wQ1
z_outlier_lower = z_wQ1 - (1.5 * z_wIQR)
z_outlier_upper = z_wQ3 + (1.5 * z_wIQR)
print("===============================")
print(z_name)
print("-------------------------------")
print("Q1     :", z_wQ1) # 2.8939
print("Median :", z_wQ2) # 3.3529
print("Q3     :", z_wQ3) # 3.8333
print("-------------------------------")
print("Outlier lower:", z_outlier_lower) # 1.4848 
print("Outlier upper:", z_outlier_upper) # 5.2424
print("===============================")

plt.figure(figsize=(5,5))
plt.hist(z, bins=70, density=1, rwidth=0.8, weights=wt)
plt.axvline(z_outlier_lower, color='k', linewidth=1)
#plt.axvline(z_wQ2, color='b', linestyle='dashed', linewidth=1)
plt.axvline(z_outlier_upper, color='k', linewidth=1)
#plt.xlim(0)
plt.xlabel(z_name, size=12)
plt.ylabel("Density", size=10)
#plt.title("Weighted Histogram: difficult")
plt.tight_layout()
plt.savefig("FIG_Histogram_difficult_raw.png")
plt.show()

