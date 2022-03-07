import pandas as pd

accuracy = 0

output = input("output: ")
correct = input("correct: ")

df_output = pd.read_csv(output)
df_correct = pd.read_csv(correct)


i = 0
j = 0

while i < len(df_output.index) and j < len(df_correct.index):
    check = True
    for r in range(1, len(df_output.columns)):
        if df_output.iloc[i, r] != df_correct.iloc[j, r]:
            check = False

    if check:
        if abs(df_output.iloc[i, 0] - df_correct.iloc[j, 0]) <= 10:
            accuracy += 1
        i += 1
        j += 1
    
    else:
        if df_output.iloc[i, 0] <= df_correct.iloc[j, 0]:
            """accuracy -= 3"""
            i += 1
        else:
            """accuracy -= 3"""
            j += 1

print("Accuracy: {0}".format(accuracy))