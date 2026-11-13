# XAndesPerf - Andes Performance Extension
# RUN: llvm-mc --verify -triple riscv64 -mattr=+xandesperf < %s

# Out of range immediates
## uimmlog2xlen/uimm6
nds.bbc t0, 64, 256 # expected-error:13 {{immediate must be an integer in the range [0, 63]}}
nds.bbs t1, 64, 256 # expected-error:13 {{immediate must be an integer in the range [0, 63]}}

## uimmlog2xlen/uimm6
nds.bfos a0, a1, 64, 3 # expected-error:18 {{immediate must be an integer in the range [0, 63]}}
nds.bfoz t0, t1, 6, 64 # expected-error:21 {{immediate must be an integer in the range [0, 63]}}

## simm19_lsb00
nds.lwugp t0, 0x3fffd # expected-error:15 {{immediate must be a multiple of 4 bytes in the range [-262144, 262140]}}
nds.lwugp t0, 0x7 # expected-error:15 {{immediate must be a multiple of 4 bytes in the range [-262144, 262140]}}
nds.lwugp t0, -0x40001 # expected-error:15 {{immediate must be a multiple of 4 bytes in the range [-262144, 262140]}}
nds.lwugp t0, -0x7 # expected-error:15 {{immediate must be a multiple of 4 bytes in the range [-262144, 262140]}}

## simm20_lsb000
nds.ldgp t0, 0x7fff9 #  expected-error:14 {{immediate must be a multiple of 8 bytes in the range [-524288, 524280]}}
nds.ldgp t0, 0x14 #  expected-error:14 {{immediate must be a multiple of 8 bytes in the range [-524288, 524280]}}
nds.ldgp t0, -0x80001 #  expected-error:14 {{immediate must be a multiple of 8 bytes in the range [-524288, 524280]}}
nds.ldgp t0, -0x14 #  expected-error:14 {{immediate must be a multiple of 8 bytes in the range [-524288, 524280]}}
nds.sdgp t0, 0x7fff9 #  expected-error:14 {{immediate must be a multiple of 8 bytes in the range [-524288, 524280]}}
nds.sdgp t0, 0x14 #  expected-error:14 {{immediate must be a multiple of 8 bytes in the range [-524288, 524280]}}
nds.sdgp t0, -0x80001 #  expected-error:14 {{immediate must be a multiple of 8 bytes in the range [-524288, 524280]}}
nds.sdgp t0, -0x14 #  expected-error:14 {{immediate must be a multiple of 8 bytes in the range [-524288, 524280]}}
