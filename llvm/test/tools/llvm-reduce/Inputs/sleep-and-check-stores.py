#!/bin/python

import time
import sys

sleep_seconds = int(sys.argv[1])
num_stores = int(sys.argv[2])
file_input = sys.argv[3]

try:
    input = open(file_input, "r")
except Exception as err:
    print(err, file=sys.stderr)
    sys.exit(1)

InterestingStores = 0
for line in input:
    if "store" in line:
        InterestingStores += 1

print("Interesting stores ", InterestingStores, " sleeping ", sleep_seconds)
time.sleep(sleep_seconds)


if InterestingStores > num_stores:
    sys.exit(0)  # interesting!

sys.exit(1)  # IR isn't interesting
