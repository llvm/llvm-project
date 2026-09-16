# RUN: "%python" "%s" "%counter"

import os
import sys

counter_file = sys.argv[1]

# A test that fails its first three attempts and passes on the fourth, so that
# it outlives the in-place retries of the first pass.
attempt = 1
if os.path.exists(counter_file):
    with open(counter_file, "r") as counter:
        attempt = int(counter.read()) + 1

with open(counter_file, "w") as counter:
    counter.write(str(attempt))

if attempt >= 4:
    sys.exit(0)

print("attempt %d failed while the machine was busy" % attempt)
sys.exit(1)
