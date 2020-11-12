import numpy as np
from matplotlib import pyplot as plt

stat=[]
for cat in ["low_risk", "medium_risk", "melanoma"]:	
	file=open(cat+"_results.txt", 'r')
	lines=file.readlines()[1:-3]
	for l in lines:
		ind=l.split(" ")[1]
		asym=l.split(" ")[3]
		stat.append((ind,asym))
		
low_risk=np.around(np.array(stat[:11]).astype(np.float),3)
medium_risk=np.around(np.array(stat[11:27]).astype(np.float),3)
melanoma=np.around(np.array(stat[27:54]).astype(np.float),3)

bins=5
plt.figure()
plt.hist(low_risk[:,0], bins, alpha=1, label='indentation_low_risk')
plt.hist(medium_risk[:,0], bins, alpha=0.5, label='indentation_medium_risk')
plt.hist(melanoma[:,0], bins, alpha=0.5, label='indentation_melanoma')
plt.legend(loc='upper right')
plt.xlabel('indentation coefficient')
plt.ylabel('absolute frequency of indentation coefficient')
plt.title("Histograms of indentation for 3 categories")
plt.grid()
plt.show()
plt.savefig("./fig/indentation.png")

plt.figure()
plt.hist(low_risk[:,1], bins, alpha=1, label='asymmetry_low_risk')
plt.hist(medium_risk[:,1], bins, alpha=0.5, label='asymmetry_medium_risk')
plt.hist(melanoma[:,1], bins, alpha=0.5, label='asymmetry_melanoma')
plt.legend(loc='upper right')
plt.xlabel('asymmetry coefficient')
plt.ylabel('absolute frequency of asymmetry coefficient')
plt.title("Histograms of asymmetry for 3 categories")
plt.grid()
plt.show()
plt.savefig("./fig/asymmetry.png")

