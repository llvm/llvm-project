import sys
f = open("spawn.txt", 'r')
g = open("simple.txt", 'r')
total1 = 0
for line in f.readlines():
	total1 += int(line[:len(line)-1])
total2 = 0
for line in g.readlines():
        total2 += int(line[:len(line)-1])
print "Spawn to serial ratio: " + str((total1*1.0)/total2)
