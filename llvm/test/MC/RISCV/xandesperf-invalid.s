# XAndesPerf - Andes Performance Extension
# RUN: llvm-mc --verify -triple riscv32 -mattr=+xandesperf < %s

# Out of range immediates
## uimmlog2xlen/uimm5
nds.bbc t0, 33, 256 # expected-error:13 {{immediate must be an integer in the range [0, 31]}}
nds.bbs t1, 33, 256 # expected-error:13 {{immediate must be an integer in the range [0, 31]}}

## uimm7
nds.beqc t0, 1024, 13 # expected-error:14 {{immediate must be an integer in the range [0, 127]}}
nds.bnec t1, -1, -13 # expected-error:14 {{immediate must be an integer in the range [0, 127]}}

## simm11_lsb0
nds.bbc t0, 7, 1024 # expected-error:16 {{immediate must be a multiple of 2 bytes in the range [-1024, 1022]}}
nds.bbs t1, 21, -1200 # expected-error:17 {{immediate must be a multiple of 2 bytes in the range [-1024, 1022]}}
nds.beqc t0, 7, 13 # expected-error:17 {{immediate must be a multiple of 2 bytes in the range [-1024, 1022]}}
nds.bnec t1, 21, -13 # expected-error:18 {{immediate must be a multiple of 2 bytes in the range [-1024, 1022]}}

## uimmlog2xlen/uimm5
nds.bfos a0, a1, 35, 3 # expected-error:18 {{immediate must be an integer in the range [0, 31]}}
nds.bfoz t0, t1, 6, 40 # expected-error:21 {{immediate must be an integer in the range [0, 31]}}

## simm18
nds.addigp t0, 0x20000 # expected-error:16 {{immediate must be an integer in the range [-131072, 131071]}}
nds.addigp t0, -0x20001 # expected-error:16 {{immediate must be an integer in the range [-131072, 131071]}}
nds.lbgp t0, 0x20000 # expected-error:14 {{immediate must be an integer in the range [-131072, 131071]}}
nds.lbugp t0, -0x20001 # expected-error:15 {{immediate must be an integer in the range [-131072, 131071]}}
nds.sbgp t0, 0x20000 # expected-error:14 {{immediate must be an integer in the range [-131072, 131071]}}
nds.sbgp t0, -0x20001 # expected-error:14 {{immediate must be an integer in the range [-131072, 131071]}}

## simm18_lsb0
nds.lhgp t0, 0x1ffff # expected-error:14 {{immediate must be a multiple of 2 bytes in the range [-131072, 131070]}}
nds.lhgp t0, 0x3 # expected-error:14 {{immediate must be a multiple of 2 bytes in the range [-131072, 131070]}}
nds.lhugp t0, -0x20001 # expected-error:15 {{immediate must be a multiple of 2 bytes in the range [-131072, 131070]}}
nds.lhugp t0, -0x3 # expected-error:15 {{immediate must be a multiple of 2 bytes in the range [-131072, 131070]}}
nds.shgp t0, 0x1ffff # expected-error:14 {{immediate must be a multiple of 2 bytes in the range [-131072, 131070]}}
nds.shgp t0, 0x3 # expected-error:14 {{immediate must be a multiple of 2 bytes in the range [-131072, 131070]}}
nds.shgp t0, -0x20001 # CHECK, CHECK-64: :[[@LINE]]:14: error: immediate must be a multiple of 2 bytes in the range [-131072, 131070]
# expected-error@-1 {{immediate must be a multiple of 2 bytes in the range [-131072, 131070]}}
nds.shgp t0, -0x3 # expected-error:14 {{immediate must be a multiple of 2 bytes in the range [-131072, 131070]}}

## simm19_lsb00
nds.lwgp t0, 0x3fffd # expected-error:14 {{immediate must be a multiple of 4 bytes in the range [-262144, 262140]}}
nds.lwgp t0, 0x7 # expected-error:14 {{immediate must be a multiple of 4 bytes in the range [-262144, 262140]}}
nds.lwgp t0, -0x40001 # expected-error:14 {{immediate must be a multiple of 4 bytes in the range [-262144, 262140]}}
nds.lwgp t0, -0x7 # expected-error:14 {{immediate must be a multiple of 4 bytes in the range [-262144, 262140]}}
nds.swgp t0, 0x3fffd # expected-error:14 {{immediate must be a multiple of 4 bytes in the range [-262144, 262140]}}
nds.swgp t0, 0x7 # expected-error:14 {{immediate must be a multiple of 4 bytes in the range [-262144, 262140]}}
nds.swgp t0, -0x40001 # expected-error:14 {{immediate must be a multiple of 4 bytes in the range [-262144, 262140]}}
nds.swgp t0, -0x7 # expected-error:14 {{immediate must be a multiple of 4 bytes in the range [-262144, 262140]}}
