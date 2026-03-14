# RUN: llvm-mc -triple=hexagon -filetype=obj %s -o - | llvm-objdump -r - | FileCheck %s

if (p0) call foo
#CHECK: R_HEX_B32_PCREL_X
#CHECK: R_HEX_B15_PCREL_X

