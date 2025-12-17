# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+s32c1i \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4
LBL0:

# CHECK-INST: xsr a3, atomctl
# CHECK: # encoding: [0x30,0x63,0x61]
xsr a3, atomctl

# CHECK-INST: xsr a3, scompare1
# CHECK: # encoding: [0x30,0x0c,0x61]
xsr a3, scompare1
