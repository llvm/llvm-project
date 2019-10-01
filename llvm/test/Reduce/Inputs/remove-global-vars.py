#!/usr/bin/env python

import sys

InterestingVar = 0

input = open(sys.argv[1], "r")
for line in input:
  i = line.find(';')
  if i >= 0:
    line = line[:i]
  if line.startswith("@interesting = global") or "@interesting" in line:
    InterestingVar += 1

if InterestingVar == 4:
  sys.exit(0) # interesting!

sys.exit(1)
