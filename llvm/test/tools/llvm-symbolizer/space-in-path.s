# REQUIRES: x86-registered-target

.type foo,@function
foo:
    nop

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o "%t space.o" -g

# Test both passing via stdin and via --obj.
# RUN: echo "\"%t space.o\" 0" > %t.input
# RUN: llvm-symbolizer < %t.input | FileCheck %s
# RUN: llvm-symbolizer --obj="%t space.o" 0 | FileCheck %s

# CHECK: foo
# CHECK-NEXT: space-in-path.s:5
