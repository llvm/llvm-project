# RUN: not llvm-mc %s -triple mips-unknown-linux-gnu -mattr=+fp64,-nooddspreg 2>&1 | \
# RUN:   FileCheck %s

  .module nooddspreg
  add.s $f1, $f2, $f4
# CHECK: :[[@LINE-1]]:9: error: -mno-odd-spreg prohibits the use of odd FPU registers

  .set oddspreg
  add.s $f1, $f2, $f4
# CHECK-NOT: :[[@LINE-1]]:{{[0-9]+}}: error: -mno-odd-spreg prohibits the use of odd FPU registers

  .set mips0
  add.s $f1, $f2, $f4
# CHECK: :[[@LINE-1]]:9: error: -mno-odd-spreg prohibits the use of odd FPU registers
