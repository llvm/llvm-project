#!/usr/bin/env python

from __future__ import print_function

import sys

# Currently any print-out from the custom tool is interpreted as a crash
# (i.e. test is still interesting)

print("Error: " + " ".join(sys.argv[1:]))

sys.exit(1)
