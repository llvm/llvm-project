# RUN: "%python" "%s" "%counter"

import os
import sys

counter_file = sys.argv[1]

# A test that fails the first time it is run and passes every time after that,
# standing in for one that only fails while other tests are running.
if os.path.exists(counter_file):
    sys.exit(0)

with open(counter_file, "w") as counter:
    counter.write("1")

print("failed while the machine was busy")
sys.exit(1)
