# XAndesPerf - Andes Performance Extension
# RUN: not llvm-mc -triple riscv64 -mattr=+xandesperf < %s 2>&1 \
# RUN:     | FileCheck %s

# Out of range immediates
## uimmlog2xlen/uimm6
nds.bbc t0, 64, 256 # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 63]
nds.bbs t1, 64, 256 # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 63]

## uimmlog2xlen/uimm6
nds.bfos a0, a1, 64, 3 # CHECK: :[[@LINE]]:18: error: immediate must be an integer in the range [0, 63]
nds.bfoz t0, t1, 6, 64 # CHECK: :[[@LINE]]:21: error: immediate must be an integer in the range [0, 63]

## simm19_lsb00
nds.lwugp t0, 0x3fffd # CHECK: :[[@LINE]]:15: error: immediate must be a multiple of 4 bytes in the range [-262144, 262140]
nds.lwugp t0, 0x7 # CHECK: :[[@LINE]]:15: error: immediate must be a multiple of 4 bytes in the range [-262144, 262140]
nds.lwugp t0, -0x40001 # CHECK: :[[@LINE]]:15: error: immediate must be a multiple of 4 bytes in the range [-262144, 262140]
nds.lwugp t0, -0x7 # CHECK: :[[@LINE]]:15: error: immediate must be a multiple of 4 bytes in the range [-262144, 262140]

## simm20_lsb000
nds.ldgp t0, 0x7fff9 #  CHECK: :[[@LINE]]:14: error: immediate must be a multiple of 8 bytes in the range [-524288, 524280]
nds.ldgp t0, 0x14 #  CHECK: :[[@LINE]]:14: error: immediate must be a multiple of 8 bytes in the range [-524288, 524280]
nds.ldgp t0, -0x80001 #  CHECK: :[[@LINE]]:14: error: immediate must be a multiple of 8 bytes in the range [-524288, 524280]
nds.ldgp t0, -0x14 #  CHECK: :[[@LINE]]:14: error: immediate must be a multiple of 8 bytes in the range [-524288, 524280]
nds.sdgp t0, 0x7fff9 #  CHECK: :[[@LINE]]:14: error: immediate must be a multiple of 8 bytes in the range [-524288, 524280]
nds.sdgp t0, 0x14 #  CHECK: :[[@LINE]]:14: error: immediate must be a multiple of 8 bytes in the range [-524288, 524280]
nds.sdgp t0, -0x80001 #  CHECK: :[[@LINE]]:14: error: immediate must be a multiple of 8 bytes in the range [-524288, 524280]
nds.sdgp t0, -0x14 #  CHECK: :[[@LINE]]:14: error: immediate must be a multiple of 8 bytes in the range [-524288, 524280]
