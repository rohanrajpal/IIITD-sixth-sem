import matplotlib.pyplot as plt

def returnrisk(w1,corrcoef):
	w2 = 1-w1
	risk = (w1**2 * 0.0016 + w2**2 * 0.0036 + 2*corrcoef*w1*w2*0.04*0.06)
	ret = w1 * 0.12 + w2 * 0.16

	return ret,risk


for corrcoef in [-1,-0.7,0,0.7]:
	x = []; y = []
	for wt in [1,0.75,0.5,0.25,0.05]:
		xp, yp = returnrisk(wt,corrcoef)
		x.append(xp)
		y.append(yp)
  
	print(x)
	print(y)
	plt.scatter(y,x,label=str(corrcoef))
plt.legend()
plt.show()
