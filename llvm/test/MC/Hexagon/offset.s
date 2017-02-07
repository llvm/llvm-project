# RUN: llvm-mc -arch=hexagon -filetype=obj %s | llvm-objdump -t - | FileCheck %s
#

sym_a:
.set sym_d, sym_a + 8
# CHECK: 00000000         .text 00000000 sym_a
# CHECK: 00000008         .text 00000000 sym_d
