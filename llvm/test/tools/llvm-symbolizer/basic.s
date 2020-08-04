# REQUIRES: x86-registered-target

foo:
    .space 10
    nop
    nop

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o -g

# Check input addresses specified on stdin.
# RUN: echo -e "0xa\n0xb" | llvm-symbolizer --obj=%t.o | FileCheck %s
# RUN: echo -e "10\n11" | llvm-symbolizer --obj=%t.o | FileCheck %s

# Check input addresses specified on the command-line.
# RUN: llvm-symbolizer 0xa 0xb --obj=%t.o | FileCheck %s
# RUN: llvm-symbolizer 10 11 --obj=%t.o | FileCheck %s

# Check --obj aliases --exe, -e
# RUN: llvm-symbolizer 0xa 0xb --exe=%t.o | FileCheck %s
# RUN: llvm-symbolizer 0xa 0xb --exe %t.o | FileCheck %s
# RUN: llvm-symbolizer 0xa 0xb -e %t.o | FileCheck %s
# RUN: llvm-symbolizer 0xa 0xb -e=%t.o | FileCheck %s
# RUN: llvm-symbolizer 0xa 0xb -e%t.o | FileCheck %s

# CHECK: basic.s:5:0
# CHECK: basic.s:6:0
