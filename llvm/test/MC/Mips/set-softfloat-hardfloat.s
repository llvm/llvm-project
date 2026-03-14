# RUN: llvm-mc %s -triple=mips-unknown-linux -mcpu=mips32r2 -mattr=+soft-float | \
# RUN:   FileCheck %s

  .set hardfloat
  add.s $f2, $f2, $f2
  sub.s $f2, $f2, $f2
  .set softfloat

# CHECK: .set hardfloat
# CHECK: add.s $f2, $f2, $f2
# CHECK: sub.s $f2, $f2, $f2
# CHECK: .set softfloat
