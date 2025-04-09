# XAndesPerf - Andes Performance Extension
# RUN: not llvm-mc -triple riscv32 -mattr=+xandesperf < %s 2>&1 \
# RUN:     | FileCheck %s
# RUN: not llvm-mc -triple riscv64 -mattr=+xandesperf < %s 2>&1 \
# RUN:     | FileCheck %s -check-prefix=CHECK-64

# Out of range immediates
## uimm5
nds.bbc t0, 33, 256 # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 31]
nds.bbs t1, 33, 256 # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 31]
nds.bbc t0, 64, 256 # CHECK-64: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 63]
nds.bbs t1, 64, 256 # CHECK-64: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 63]

## uimm7
nds.beqc t0, 1024, 13 # CHECK, CHECK-64: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 127]
nds.bnec t1, -1, -13 # CHECK, CHECK-64: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 127]

## simm11_lsb0
nds.bbc t0, 7, 1024 # CHECK, CHECK-64: :[[@LINE]]:16: error: immediate must be a multiple of 2 bytes in the range [-1024, 1022]
nds.bbs t1, 21, -1200 # CHECK, CHECK-64: :[[@LINE]]:17: error: immediate must be a multiple of 2 bytes in the range [-1024, 1022]
nds.beqc t0, 7, 13 # CHECK, CHECK-64: :[[@LINE]]:17: error: immediate must be a multiple of 2 bytes in the range [-1024, 1022]
nds.bnec t1, 21, -13 # CHECK, CHECK-64: :[[@LINE]]:18: error: immediate must be a multiple of 2 bytes in the range [-1024, 1022]

## uimmlog2xlen
nds.bfos a0, a1, 35, 3 # CHECK: :[[@LINE]]:18: error: immediate must be an integer in the range [0, 31]
nds.bfoz t0, t1, 6, 40 # CHECK: :[[@LINE]]:21: error: immediate must be an integer in the range [0, 31]
nds.bfos a0, a1, 64, 3 # CHECK-64: :[[@LINE]]:18: error: immediate must be an integer in the range [0, 63]
nds.bfoz t0, t1, 6, 64 # CHECK-64: :[[@LINE]]:21: error: immediate must be an integer in the range [0, 63]

## simm18
nds.addigp t0, 0x20000 # CHECK, CHECK-64: :[[@LINE]]:16: error: immediate must be an integer in the range [-131072, 131071]
nds.addigp t0, -0x20001 # CHECK, CHECK-64: :[[@LINE]]:16: error: immediate must be an integer in the range [-131072, 131071]
nds.lbgp t0, 0x20000 # CHECK, CHECK-64: :[[@LINE]]:14: error: immediate must be an integer in the range [-131072, 131071]
nds.lbugp t0, -0x20001 # CHECK, CHECK-64: :[[@LINE]]:15: error: immediate must be an integer in the range [-131072, 131071]
nds.sbgp t0, 0x20000 # CHECK, CHECK-64: :[[@LINE]]:14: error: immediate must be an integer in the range [-131072, 131071]
nds.sbgp t0, -0x20001 # CHECK, CHECK-64: :[[@LINE]]:14: error: immediate must be an integer in the range [-131072, 131071]

## simm18_lsb0
nds.lhgp t0, 0x1ffff # CHECK, CHECK-64: :[[@LINE]]:14: error: immediate must be a multiple of 2 bytes in the range [-131072, 131070]
nds.lhgp t0, 0x3 # CHECK, CHECK-64: :[[@LINE]]:14: error: immediate must be a multiple of 2 bytes in the range [-131072, 131070]
nds.lhugp t0, -0x20001 # CHECK, CHECK-64: :[[@LINE]]:15: error: immediate must be a multiple of 2 bytes in the range [-131072, 131070]
nds.lhugp t0, -0x3 # CHECK, CHECK-64: :[[@LINE]]:15: error: immediate must be a multiple of 2 bytes in the range [-131072, 131070]
nds.shgp t0, 0x1ffff # CHECK, CHECK-64: :[[@LINE]]:14: error: immediate must be a multiple of 2 bytes in the range [-131072, 131070]
nds.shgp t0, 0x3 # CHECK, CHECK-64: :[[@LINE]]:14: error: immediate must be a multiple of 2 bytes in the range [-131072, 131070]
nds.shgp t0, -0x20001 # CHECK, CHECK-64: :[[@LINE]]:14: error: immediate must be a multiple of 2 bytes in the range [-131072, 131070]
nds.shgp t0, -0x3 # CHECK, CHECK-64: :[[@LINE]]:14: error: immediate must be a multiple of 2 bytes in the range [-131072, 131070]

## simm19_lsb00
nds.lwgp t0, 0x3fffd # CHECK, CHECK-64: :[[@LINE]]:14: error: immediate must be a multiple of 4 bytes in the range [-262144, 262140]
nds.lwgp t0, 0x7 # CHECK, CHECK-64: :[[@LINE]]:14: error: immediate must be a multiple of 4 bytes in the range [-262144, 262140]
nds.lwgp t0, -0x40001 # CHECK, CHECK-64: :[[@LINE]]:14: error: immediate must be a multiple of 4 bytes in the range [-262144, 262140]
nds.lwgp t0, -0x7 # CHECK, CHECK-64: :[[@LINE]]:14: error: immediate must be a multiple of 4 bytes in the range [-262144, 262140]
nds.swgp t0, 0x3fffd # CHECK, CHECK-64: :[[@LINE]]:14: error: immediate must be a multiple of 4 bytes in the range [-262144, 262140]
nds.swgp t0, 0x7 # CHECK, CHECK-64: :[[@LINE]]:14: error: immediate must be a multiple of 4 bytes in the range [-262144, 262140]
nds.swgp t0, -0x40001 # CHECK, CHECK-64: :[[@LINE]]:14: error: immediate must be a multiple of 4 bytes in the range [-262144, 262140]
nds.swgp t0, -0x7 # CHECK, CHECK-64: :[[@LINE]]:14: error: immediate must be a multiple of 4 bytes in the range [-262144, 262140]
