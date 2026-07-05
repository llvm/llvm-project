import sys

InterestingBBs = 0
input = open(sys.argv[1], "r")
for line in input:
    i = line.find(";")
    if i >= 0:
        line = line[:i]
    if line.startswith("interesting") or "%interesting" in line:
        InterestingBBs += 1

if InterestingBBs == 6:
    sys.exit(0)  # interesting!

sys.exit(1)  # IR isn't interesting
