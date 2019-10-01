# REQUIRES: x86-registered-target

## If a process spawns llvm-symbolizer, and wishes to feed it addresses one at a
## time, llvm-symbolizer needs to flush its output after each input has been
## processed or the parent process will not be able to read the output and may
## deadlock. This test runs a script that simulates this situation for both a
## a good and bad input.

foo:
    nop

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o -g
# RUN: %python %p/Inputs/flush-output.py llvm-symbolizer %t.o \
# RUN:   | FileCheck %s

# CHECK: flush-output.s:10
# CHECK: bad
