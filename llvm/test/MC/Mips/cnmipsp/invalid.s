# Instructions that are invalid.
#
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -mcpu=octeon+ 2>%t1
# RUN: FileCheck %s < %t1

saa  $2         # CHECK: :[[@LINE]]:1: error: too few operands for instruction
saa  $2, $5, $6 # CHECK: :[[@LINE]]:12: error: unexpected token in argument list

saad $2         # CHECK: :[[@LINE]]:1: error: too few operands for instruction
saad $2, $5, $6 # CHECK: :[[@LINE]]:12: error: unexpected token in argument list
