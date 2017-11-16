# RUN: not llvm-mc -arch=mips -mcpu=mips32r2 -mattr=+mt < %s 2>%t1
# RUN: FileCheck %s < %t1
  mftr 0($4), $5, 0, 0, 0 # CHECK: error: unexpected token in argument list
  mttr 0($4), $5, 0, 0, 0 # CHECK: error: unexpected token in argument list
