# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+threadptr \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4
LBL0:

# CHECK-INST: rur a3, threadptr
# CHECK: encoding: [0x70,0x3e,0xe3]
rur a3, threadptr

# CHECK-INST: rur a3, threadptr
# CHECK: encoding: [0x70,0x3e,0xe3]
rur a3, 231

# CHECK-INST: rur a3, threadptr
# CHECK: encoding: [0x70,0x3e,0xe3]
rur.threadptr a3

# CHECK-INST: wur a3, threadptr
# CHECK: encoding: [0x30,0xe7,0xf3]
wur a3, threadptr
