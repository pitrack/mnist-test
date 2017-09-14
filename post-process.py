import csv
import os
import numpy
import matplotlib.pyplot as plt
for f in os.listdir('.'):
    if os.path.isfile(f) and f.endswith('.csv'):
        print(f)
        result = []
        with open(f,'r') as csvfile:
            reader=csv.reader(csvfile)
            base = None
            for row in reader:
                if base is None:
                    base = row
                #print("{},{}".format(float(row[0])-float(base[0]),(int(row[1])-int(base[1]))/1024/1024))
                result.append((float(row[0])-float(base[0]),(int(row[1])-int(base[1]))/1024/1024))
        data = [x[1] for x in result]
        plt.plot(range(len(data)),data)
        plt.savefig(f[:-4]+".png")