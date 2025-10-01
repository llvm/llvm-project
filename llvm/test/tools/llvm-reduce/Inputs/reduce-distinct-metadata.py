# Helper script for distinct metadata reduction test

import sys
import re

input = open(sys.argv[1], "r").read().splitlines()

depth_map = {"0": 1, "1": 3, "2": 3, "3": 2, "4": 1}


for i in range(len(depth_map)):
    counter = 0
    for line in input:
        if re.match(rf".*interesting_{i}.*", line) != None:
            counter += 1
    if counter != depth_map[str(i)]:
        sys.exit(1)

sys.exit(0)
