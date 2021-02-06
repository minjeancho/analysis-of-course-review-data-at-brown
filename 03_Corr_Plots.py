import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import seaborn
import scipy

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

# === Weighted Percentile Calculator =======================
def weighted_percentile(array, percentile, sample_weights):
    # percentile range: 0-100
    repeated_samples = np.repeat(array, sample_weights)
    percentile = np.percentile(repeated_samples, percentile)
    return percentile

# === Weighted Correlation Matrix Calculator ==============
def calculate_weighted_corr_matrix(values, weights):
    # values from dataframe: df.values
    weighted_covariance_matrix  = np.cov(values.T, fweights = weights)
    weighted_correlation_matrix = np.copy(weighted_covariance_matrix) 
    for i in range(values.shape[1]): # shape = (2179, 4)
        for j in range(values.shape[1]):
            weighted_variance_x = weighted_covariance_matrix[i][i]
            weighted_variance_y = weighted_covariance_matrix[j][j]
            denominator = (weighted_variance_x * weighted_variance_y)**(1/2)
            weighted_correlation_matrix[i][j] = weighted_correlation_matrix[i][j]/denominator
    return weighted_correlation_matrix

def partial_correlation(r_xy, r_xz, r_yz):
    r_xy_z = (r_xy - r_xz*r_yz) / (((1 - r_xz**2)**(1/2)) * ((1 - r_yz**2)**(1/2)))    
    return r_xy_z

def pearson_r_p_value(r, n_samples):
    # two-tailed p-value based on t-distribution (d_freedom = n-2)
    d_freedom = n_samples - 2
    t = (r*np.sqrt(n_samples-2))/np.sqrt(1-(r**2))
    p_value = scipy.stats.t.sf(np.abs(t), d_freedom)*2
    return p_value

def partial_r_p_value(r, n_samples):
    # two-tailed p-value based on t-distribution (d_freedom = n-2-k)
    d_freedom = n_samples - 3 # k = 1 (no. of controlled variables)
    t = r*np.sqrt(d_freedom/(1-r*2))
    p_value = scipy.stats.t.sf(np.abs(t), d_freedom)*2
    return p_value

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    # REFERENCE: matplotlib example (modified)
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    cov = np.cov(x, y, fweights=wt)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # print(pearson)
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)
    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def show_correaltion_plot(x, y, x_name, y_name, weights, save_file=1, prefix="BROWN"):
    # REFERENCE: matplotlib example (modified)
    # figure shape parameters
    mu = 2, 4
    scale = 3, 5
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.07]
    rect_histy = [left + width + spacing, bottom, 0.07, height]
    # start with a rectangular Figure
    plt.figure(figsize=(8, 8))
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    #ax_histx = plt.axes(rect_histx)
    #ax_histx.tick_params(direction='in', color='w', labelbottom=False, labelcolor='w')
    #ax_histy = plt.axes(rect_histy)
    #ax_histy.tick_params(direction='in', color='w', labelleft=False, labelcolor='w')
    #fig, ax = plt.subplots(1, figsize=(9, 3))
    rel_weights = weights / np.sum(weights)
    ax_scatter.scatter(x,y,s=rel_weights*len(weights)*20,
                       edgecolors='white', linewidths=0.5)
    #ax.scatter(x, y, s=0.5)
    #ax.axvline(c='grey', lw=1)
    #ax.axhline(c='grey', lw=1)
    confidence_ellipse(x, y, ax_scatter, edgecolor='black', linestyle='dotted')
    #ax.scatter(mu[0], mu[1], c='red', s=3)
    #ax.set_title("title")
    ax_scatter.set_xlabel(x_name, size=12)
    ax_scatter.set_ylabel(y_name, size=12)
    # now determine nice limits by hand:
    binwidth = 0.25
    lim = np.ceil(np.abs([x, y]).max() / binwidth) * binwidth
    x_min = np.min(x)
    x_max = np.max(x)
    x_off = (x_max - x_min)/20
    y_min = np.min(y)
    y_max = np.max(y)
    y_off = (y_max - y_min)/20
    ax_scatter.set_xlim((x_min-x_off, x_max+x_off))
    ax_scatter.set_ylim((y_min-y_off, y_max+y_off))
    #bins = np.arange(-lim, lim + binwidth, binwidth)
    #ax_histx.hist(x, weights=weights, bins = 70, density=1, rwidth=0.8,) #, bins=bins)
    #ax_histy.hist(y, weights=weights, bins = 70, density=1, rwidth=0.8,
    #              orientation = 'horizontal') #, bins=bins, orientation='horizontal')
    #ax_histx.set_xlim(ax_scatter.get_xlim())
    #ax_histy.set_ylim(ax_scatter.get_ylim())
    #plt.text(-10, -10, "Pearson r:") #, str(r_12))
    #plt.tight_layout()
    if save_file == 1:
        file_name = "FIG_scatter_" + prefix + "_" + x_name + "_" + y_name + ".png"
        plt.savefig(file_name)
    plt.show()    

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

# ==================================
x_name = "minhours"
y_name = "grade"
z_name = "difficult"
weight_name = "class_size"
# === REMOVE OUTLIERS ==============
x = df_BROWN[x_name]
y = df_BROWN[y_name]
z = df_BROWN[z_name]
wt = df_BROWN[weight_name] 

x_wQ1 = weighted_percentile(x, 25, wt)
x_wQ3 = weighted_percentile(x, 75, wt)
x_wIQR = x_wQ3 - x_wQ1
x_outlier_lower = x_wQ1 - (1.5 * x_wIQR)
x_outlier_upper = x_wQ3 + (1.5 * x_wIQR)

y_wQ1 = weighted_percentile(y, 25, wt)
y_wQ3 = weighted_percentile(y, 75, wt)
y_wIQR = y_wQ3 - y_wQ1
y_outlier_lower = y_wQ1 - (1.5 * y_wIQR)
y_outlier_upper = y_wQ3 + (1.5 * y_wIQR)

z_wQ1 = weighted_percentile(z, 25, wt)
z_wQ3 = weighted_percentile(z, 75, wt)
z_wIQR = z_wQ3 - z_wQ1
z_outlier_lower = z_wQ1 - (1.5 * z_wIQR)
z_outlier_upper = z_wQ3 + (1.5 * z_wIQR)

x = x.values
y = y.values
z = z.values
wt = wt.values

for i in range(x.shape[0]):
    if   x[i] > x_outlier_upper: x[i] = np.nan
    elif x[i] < x_outlier_lower: x[i] = np.nan

for i in range(y.shape[0]):
    if   y[i] > y_outlier_upper: y[i] = np.nan
    elif y[i] < y_outlier_lower: y[i] = np.nan
    #elif y[i] == 4.0: y[i] = np.nan

for i in range(z.shape[0]):
    if   z[i] > z_outlier_upper: z[i] = np.nan
    elif z[i] < z_outlier_lower: z[i] = np.nan    

df = pd.DataFrame({x_name: x, y_name: y, z_name: z, weight_name: wt})
df = df.dropna(axis=0)
print(df)

# === Outlier removed distribution
x  = df['minhours'  ].values
y  = df['grade'     ].values
z  = df['difficult' ].values
wt = df['class_size'].values

values  = df[['minhours', 'grade', 'difficult']].values
weights = df['class_size'].values
n_samples = df.shape[0]
w_corr_matrix = calculate_weighted_corr_matrix(values, weights)
r_xy = w_corr_matrix[0][1]
r_xz = w_corr_matrix[0][2]
r_yz = w_corr_matrix[1][2]
r_xy_z = partial_correlation(r_xy, r_xz, r_yz)
# p-values for pearson r
p_r_xy = pearson_r_p_value(r_xy, n_samples)
p_r_xz = pearson_r_p_value(r_xz, n_samples)
p_r_yz = pearson_r_p_value(r_yz, n_samples)
# p-value for partial r
p_r_xy_z = partial_r_p_value(r_xy_z, n_samples)
print()
print("===============================================")
print("Correaltion Matrix [BEFORE boxcox transformation")
print("          number of samples:", n_samples)
print("-----------------------------------------------")
print("W Corr[minhours,  grade]     : %.2f, p-value: %.4f" % (r_xy, p_r_xy))
print("W Corr[minhours,  difficult']: %.2f, p-value: %.4f" % (r_xz, p_r_xz))
print("W Corr[difficult, grade]     : %.2f, p-value: %.4f" % (r_yz, p_r_yz))
print("-----------------------------------------------")
print("P Corr[minhours, grade | difficult]: %.2f, p-value: %.4f" % (r_xy_z, p_r_xy_z))
print("===============================================")
print()

show_correaltion_plot(x, y, x_name, y_name, wt, save_file=1)
show_correaltion_plot(z, x, z_name, x_name, wt, save_file=1)
show_correaltion_plot(z, y, z_name, y_name, wt, save_file=1)

# === NORMALIZING TRANSFORMATION (BOX-COX)
x = scipy.stats.boxcox(x)[0]
y = scipy.stats.boxcox(y)[0]
z = scipy.stats.boxcox(z)[0]

df_boxcox = pd.DataFrame({x_name: x, y_name: y, z_name: z, weight_name: wt})

values  = df_boxcox[['minhours', 'grade', 'difficult']].values
weights = df_boxcox['class_size'].values
n_samples = df_boxcox.shape[0]
w_corr_matrix = calculate_weighted_corr_matrix(values, weights)
r_xy = w_corr_matrix[0][1]
r_xz = w_corr_matrix[0][2]
r_yz = w_corr_matrix[1][2]
r_xy_z = partial_correlation(r_xy, r_xz, r_yz)
# p-values for pearson r
p_r_xy = pearson_r_p_value(r_xy, n_samples)
p_r_xz = pearson_r_p_value(r_xz, n_samples)
p_r_yz = pearson_r_p_value(r_yz, n_samples)
p_r_xy_z = partial_r_p_value(r_xy_z, n_samples)
print()
print("===============================================")
print("Correaltion Matrix [AFTER boxcox transformation")
print("          number of samples:", n_samples)
print("-----------------------------------------------")
print("W Corr[minhours,  grade]     : %.2f, p-value: %.4f" % (r_xy, p_r_xy))
print("W Corr[minhours,  difficult']: %.2f, p-value: %.4f" % (r_xz, p_r_xz))
print("W Corr[difficult, grade]     : %.2f, p-value: %.4f" % (r_yz, p_r_yz))
print("-----------------------------------------------")
print("P Corr[minhours, grade | difficult]: %.2f, p-value: %.4f" % (r_xy_z, p_r_xy_z))
print("===============================================")
print()

show_correaltion_plot(x, y, x_name, y_name, wt, save_file=1, prefix="BROWN_BoxCoX")
show_correaltion_plot(z, x, z_name, x_name, wt, save_file=1, prefix="BROWN_BoxCoX")
show_correaltion_plot(z, y, z_name, y_name, wt, save_file=1, prefix="BROWN_BoxCoX")

