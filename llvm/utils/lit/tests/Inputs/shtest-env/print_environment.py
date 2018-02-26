#!/usr/bin/env python

import os

sorted_environment = sorted(os.environ.items())

for name,value in sorted_environment:
    print name,'=',value
