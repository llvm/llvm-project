## Show that when "CODE" is used with an address, it forces the found location
## to be symbolized as a function (this is the default).
# REQUIRES: x86-registered-target
# RUN: llvm-mc -g -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: llvm-symbolizer "CODE 0x1" "CODE 0x2" --obj=%t.o > %t.code
# RUN: llvm-symbolizer 0x1 0x2 --obj=%t.o > %t.default
# RUN: cmp %t.code %t.default
# RUN: FileCheck %s --input-file=%t.code -DFILE=%s --implicit-check-not={{.}}

# CHECK:      f1
f1:
    nop
# CHECK-NEXT: [[FILE]]:[[@LINE+1]]:0
    ret
# CHECK-EMPTY:
# CHECK-NEXT: f2
f2:
# CHECK-NEXT: [[FILE]]:[[@LINE+1]]:0
    ret
