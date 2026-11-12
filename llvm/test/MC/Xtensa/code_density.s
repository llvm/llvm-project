# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+density \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4
LBL0:

# Instruction format RRRN
# CHECK-INST: add.n a2, a3, a4
# CHECK: encoding: [0x4a,0x23]
add.n a2, a3, a4

# Instruction format RRRN
# CHECK-INST: addi.n a2, a3, 3
# CHECK: encoding: [0x3b,0x23]
addi.n a2, a3, 3

# Instruction format RRRN
# CHECK-INST: addi.n a2, a3, -1
# CHECK: encoding: [0x0b,0x23]
addi.n a2, a3, -1

# Instruction format RI6
# CHECK-INST:  beqz.n  a3, LBL1
# CHECK: encoding: [0x8c'A',0x03'A']
beqz.n  a3, LBL1

# Instruction format RI6
# CHECK-INST:  bnez.n  a3, LBL1
# CHECK: encoding: [0xcc'A',0x03'A']
bnez.n  a3, LBL1

# Instruction format RRRN
# CHECK-INST: ill.n
# CHECK: encoding: [0x6d,0xf0]
ill.n

# Instruction format RRRN
# CHECK-INST: l32i.n a2, a3, 12
# CHECK: encoding: [0x28,0x33]
l32i.n a2, a3, 12

# Instruction format RRRN
# CHECK-INST: mov.n a2, a3
# CHECK: encoding: [0x2d,0x03]
mov.n a2, a3

# Instruction format RI7
# CHECK-INST: movi.n a2, -32
# CHECK: encoding: [0x6c,0x02]
movi.n a2, -32

# Instruction format RRRN
# CHECK-INST: nop.n
# CHECK: encoding: [0x3d,0xf0]
nop.n

# Instruction format RRRN
# CHECK-INST: ret.n
# CHECK: encoding: [0x0d,0xf0]
ret.n

# Instruction format RRRN
# CHECK-INST: s32i.n a2, a3, 12
# CHECK: encoding: [0x29,0x33]
s32i.n a2, a3, 12

.align	4
LBL1:
