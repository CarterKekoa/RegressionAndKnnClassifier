import random
from tabulate import tabulate

def get_rand_rows(table, num_rows):
    rand_rows = []
    for i in range(num_rows):
        rand_rows.append(table.data[random.randint(0,len(table.data))-1])
    return rand_rows

def get_rating(mpg):
    if mpg < 14:
        return 1
    elif mpg < 15:
        return 2
    elif mpg < 17:
        return 3
    elif mpg < 20:
        return 4
    elif mpg < 24:
        return 5
    elif mpg < 27:
        return 6
    elif mpg < 31:
        return 7
    elif mpg < 37:
        return 8
    elif mpg < 45:
        return 9
    return 10

def count_correct(predicted, expected):
    count = 0
    assert len(predicted) == len(expected)
    for i in predicted:
        if predicted[i] == expected[i]:
            count += 1
    return count

def add_conf_stats(matrix):
    del matrix[0]
    for i,row in enumerate(matrix):
        row[0] = i+1
        row.append(sum(row))
        row.append(round(row[i+1]/row[-1]*100,2))

def print_tabulate(table, headers):
    print(tabulate(table, headers, tablefmt="rst"))
