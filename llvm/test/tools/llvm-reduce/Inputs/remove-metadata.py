import sys

input = open(sys.argv[1], "r")
for line in input:
  if "!interesting" in line:
    sys.exit(0)

sys.exit(1)
