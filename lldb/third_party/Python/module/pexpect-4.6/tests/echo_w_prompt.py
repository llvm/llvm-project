# -*- coding: utf-8 -*-
from __future__ import print_function

try:
    raw_input
except NameError:
    raw_input = input

while True:
    try:
        a = raw_input('<in >')
    except EOFError:
        print('<eof>')
        break
    print('<out>', a, sep='')
