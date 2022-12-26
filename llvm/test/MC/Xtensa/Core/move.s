# RUN: llvm-mc %s -triple=xtensa -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s


.align	4
LBL0:

# Instruction format RRR
# CHECK-INST: moveqz a2, a3, a4
# CHECK: encoding: [0x40,0x23,0x83]
moveqz a2, a3, a4

# Instruction format RRR
# CHECK-INST: movgez a3, a11, a12
# CHECK: encoding: [0xc0,0x3b,0xb3]
movgez a3, a11, a12

# Instruction format RRI8
# CHECK-INST: movi a1, -2048
# CHECK: encoding: [0x12,0xa8,0x00]
movi a1, -2048

# Instruction format RRR
# CHECK-INST: movltz a7, a8, a9
# CHECK: encoding: [0x90,0x78,0xa3]
movltz a7, a8, a9

# Instruction format RRR
# CHECK-INST: movnez a10, a11, a12
# CHECK: encoding: [0xc0,0xab,0x93]
movnez a10, a11, a12
