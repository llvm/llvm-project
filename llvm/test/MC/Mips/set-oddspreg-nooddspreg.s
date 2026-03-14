# RUN: llvm-mc %s -triple=mips-unknown-linux -mcpu=mips32r2 -mattr=+nooddspreg | \
# RUN:   FileCheck %s

  .set oddspreg
  sub.s $f1, $f2, $f2
  .set nooddspreg

# CHECK: .set oddspreg
# CHECK: sub.s $f1, $f2, $f2
# CHECK: .set nooddspreg
