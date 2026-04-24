# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-zvvmm %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

# SEW=32, LMUL=1, ta, ma, lambda=4, altfmt_A=1, altfmt_B=1.
li t1, 0x3c000000000000d0
vsetvl t0, a0, t1
# CHECK-INST: vsetvl t0, a0, t1
# CHECK-ENCODING: [0xd7,0x72,0x65,0x80]
vmmacc.vv v8, v4, v20
# CHECK-INST-NEXT: vmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x42,0xe3]
