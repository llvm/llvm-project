# RUN: not llvm-mc %s -triple=mips-unknown-linux -mcpu=mips32 -mattr=+nooddspreg 2>%t1
# RUN: FileCheck %s < %t1

  .set oddspreg
  sub.s $f1, $f2, $f2
  # CHECK-NOT: :[[@LINE-1]]:{{[0-9]+}}: error: -mno-odd-spreg prohibits the use of odd FPU registers

  .set nooddspreg
  sub.s $f1, $f2, $f2
  # CHECK: :[[@LINE-1]]:9: error: -mno-odd-spreg prohibits the use of odd FPU registers
