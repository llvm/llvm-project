#!/usr/bin/env python

import sys

sys.stdout.write(b"a line with bad encoding: \xc2.")
sys.stdout.flush()
