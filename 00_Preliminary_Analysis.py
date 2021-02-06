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

value_array = df_BROWN[features].values
corr_matrix = np.corrcoef(value_array.T)

seaborn.set(font_scale=1)
plt.figure(figsize=(8,8))
heat_map = seaborn.heatmap(corr_matrix, vmin=-1, vmax=1, cmap="RdYlBu",
                           cbar=False, annot=True, square=True,
                           fmt='.2f', annot_kws={'size':9},
                           yticklabels=features, xticklabels=features)
plt.tight_layout()
#plt.title("Preliminary Correlation Analysis", size=12)
plt.savefig("FIG_preliminary_corr_analysis.png")
plt.show()




