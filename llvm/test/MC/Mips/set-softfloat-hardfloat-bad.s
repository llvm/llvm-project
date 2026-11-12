# RUN: not llvm-mc %s -triple=mips-unknown-linux -mcpu=mips32 -mattr=+soft-float 2>%t1
# RUN: FileCheck %s < %t1

  .set hardfloat
  add.s $f2, $f2, $f2
  # CHECK-NOT: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  sub.s $f2, $f2, $f2
  # CHECK-NOT: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled

  .set softfloat
  add.s $f2, $f2, $f2
  # CHECK: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
  sub.s $f2, $f2, $f2
  # CHECK: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled
