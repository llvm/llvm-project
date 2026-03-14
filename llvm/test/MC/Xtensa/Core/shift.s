# RUN: llvm-mc %s -triple=xtensa -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s


.align	4
LBL0:

# Instruction format RRR
# CHECK-INST: extui a1, a2, 7, 8
# CHECK: encoding: [0x20,0x17,0x74]
extui a1, a2, 7, 8

# Instruction format RRR
# CHECK-INST: sll a10, a11
# CHECK: encoding: [0x00,0xab,0xa1]
sll a10, a11

# Instruction format RRR
# CHECK-INST: slli a5, a1, 15
# CHECK: encoding: [0x10,0x51,0x11]
slli a5, a1, 15

# Instruction format RRR
# CHECK-INST: sra a12, a3
# CHECK: encoding: [0x30,0xc0,0xb1]
sra a12, a3

# Instruction format RRR
# CHECK-INST: srai a8, a5, 0
# CHECK: encoding: [0x50,0x80,0x21]
srai a8, a5, 0

# Instruction format RRR
# CHECK-INST: src a3, a4, a5
# CHECK: encoding: [0x50,0x34,0x81]
src a3, a4, a5

# Instruction format RRR
# CHECK-INST: srl a6, a7
# CHECK: encoding: [0x70,0x60,0x91]
srl a6, a7

# Instruction format RRR
# CHECK-INST: srli a3, a4, 8
# CHECK: encoding: [0x40,0x38,0x41]
srli a3, a4, 8

# Instruction format RRR
# CHECK-INST: ssa8l a14
# CHECK: encoding: [0x00,0x2e,0x40]
ssa8l a14

# Instruction format RRR
# CHECK-INST: ssai 31
# CHECK: encoding: [0x10,0x4f,0x40]
ssai 31

# Instruction format RRR
# CHECK-INST: ssl a0
# CHECK: encoding: [0x00,0x10,0x40]
ssl a0

# Instruction format RRR
# CHECK-INST: ssr a2
# CHECK: encoding: [0x00,0x02,0x40]
ssr a2
