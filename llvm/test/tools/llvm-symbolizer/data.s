## Show that when "DATA" is used with an address, it forces the found location
## to be symbolized as data.
# REQUIRES: x86-registered-target
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: llvm-symbolizer "DATA 0x2" "DATA 0x8" --obj=%t.o | \
# RUN:   FileCheck %s -DFILE=%s --implicit-check-not={{.}}

# CHECK:      d1
# CHECK-NEXT: 0 8
# CHECK-EMPTY:
# CHECK-NEXT: d2
# CHECK-NEXT: 8 4

d1:
    .quad 0x1122334455667788
    .size d1, 8

d2:
    .long 0x99aabbcc
    .size d2, 4
