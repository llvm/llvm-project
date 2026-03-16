# RUN: llvm-mc %s -triple=xtensa -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

# Test that .set symbolic constants can be used as b4const/b4constu
# immediates in branch instructions.

.set CONST_B4, 4
.set CONST_B4_NEG, -1
.set CONST_B4U, 65536

.align	4
LBL0:

# CHECK-INST:  bnei    a6, 4, LBL0
# CHECK: encoding: [0x66,0x46,A]
bnei a6, CONST_B4, LBL0

# CHECK-INST:  beqi    a1, 4, LBL0
# CHECK: encoding: [0x26,0x41,A]
beqi a1, CONST_B4, LBL0

# CHECK-INST:  bgei    a11, -1, LBL0
# CHECK: encoding: [0xe6,0x0b,A]
bgei a11, CONST_B4_NEG, LBL0

# CHECK-INST:  blti    a0, 4, LBL0
# CHECK: encoding: [0xa6,0x40,A]
blti a0, CONST_B4, LBL0

# CHECK-INST:  bgeui   a7, 65536, LBL0
# CHECK: encoding: [0xf6,0x17,A]
bgeui a7, CONST_B4U, LBL0

# CHECK-INST:  bltui   a7, 4, LBL0
# CHECK: encoding: [0xb6,0x47,A]
bltui a7, CONST_B4, LBL0
