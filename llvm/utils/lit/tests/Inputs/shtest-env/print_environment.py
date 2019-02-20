#!/usr/bin/env python

from __future__ import print_statement
import os

sorted_environment = sorted(os.environ.items())

for name,value in sorted_environment:
    print(name,'=',value)
