#!/usr/bin/env python3

"""A test case generation script.

Generate check lines for all known TargetLibraryInfo functions

"""

import sys
import os
import subprocess

if len(sys.argv) < 2:
    print("usage: " + sys.argv[0] + " target-triple-name > target-triple-name.test")
    exit(1)

triple = sys.argv[1]
process = subprocess.Popen(
    "llvm-tli-checker --dump-tli --triple=" + triple,
    shell=True,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,
    universal_newlines=True)

print("# RUN: llvm-tli-checker --dump-tli --triple=" + triple + " | FileCheck %s\n")

First = True
for line in process.stdout.readlines():
    if First:
        print("CHECK: " + line.strip());
        First = False
    else:
        print("CHECK-NEXT: " + line.strip());
