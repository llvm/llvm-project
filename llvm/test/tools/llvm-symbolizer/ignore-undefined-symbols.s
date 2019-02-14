# REQUIRES: x86-registered-target
# RUN: llvm-mc --filetype=obj --triple=x86_64-pc-linux %s -o %t.o -g
# RUN: llvm-symbolizer --obj=%t.o 0 | FileCheck %s --implicit-check-not=bar

# CHECK:      foo
# CHECK-NEXT: ignore-undefined-symbols.s:12:0

.type bar,@function
.type foo,@function
.global foo
foo:
    call bar
