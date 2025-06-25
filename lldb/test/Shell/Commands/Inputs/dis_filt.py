#! /usr/bin/env python

import sys

for line in sys.stdin:
    if '0940003f 00200020' in line and '<unknown>' in line:
        line = line.replace('<unknown>', 'Fake64')
    print(line, end="")
