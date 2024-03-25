import sys

InterestingInstructions = 0

input = open(sys.argv[1], "r")
for line in input:
    i = line.find(";")
    if i >= 0:
        line = line[:i]
    if "%interesting" in line:
        InterestingInstructions += 1
    print(InterestingInstructions)

if InterestingInstructions == 5:
    sys.exit(0)  # interesting!

sys.exit(1)
