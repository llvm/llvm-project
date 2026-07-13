# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+exception \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4
LBL0:

# Instruction format RRR
# CHECK-INST: excw
# CHECK: encoding: [0x80,0x20,0x00]
excw

# Instruction format RRR
# CHECK-INST: syscall
# CHECK: encoding: [0x00,0x50,0x00]
syscall

# Instruction format RRR
# CHECK-INST: rfe
# CHECK: encoding: [0x00,0x30,0x00]
rfe

# Instruction format RRR
# CHECK-INST: rfde
# CHECK: encoding: [0x00,0x32,0x00]
rfde

# Instruction format RRR
# CHECK-INST: xsr a2, epc1
# CHECK: encoding: [0x20,0xb1,0x61]
xsr a2, epc1

# Instruction format RRR
# CHECK-INST: xsr a2, epc1
# CHECK: encoding: [0x20,0xb1,0x61]
xsr.epc1 a2

# Instruction format RRR
# CHECK-INST: xsr a2, epc1
# CHECK: encoding: [0x20,0xb1,0x61]
xsr a2, 177

# Instruction format RRR
# CHECK-INST: xsr a2, excsave1
# CHECK: encoding: [0x20,0xd1,0x61]
xsr a2, excsave1

# Instruction format RRR
# CHECK-INST: xsr a2, excsave1
# CHECK: encoding: [0x20,0xd1,0x61]
xsr.excsave1 a2

# Instruction format RRR
# CHECK-INST: xsr a2, excsave1
# CHECK: encoding: [0x20,0xd1,0x61]
xsr a2, 209

# Instruction format RRR
# CHECK-INST: xsr a2, exccause
# CHECK: encoding: [0x20,0xe8,0x61]
xsr a2, exccause

# Instruction format RRR
# CHECK-INST: xsr a2, exccause
# CHECK: encoding: [0x20,0xe8,0x61]
xsr.exccause a2

# Instruction format RRR
# CHECK-INST: xsr a2, exccause
# CHECK: encoding: [0x20,0xe8,0x61]
xsr a2, 232

# Instruction format RRR
# CHECK-INST: xsr a2, excvaddr
# CHECK: encoding: [0x20,0xee,0x61]
xsr a2, excvaddr

# Instruction format RRR
# CHECK-INST: xsr a2, excvaddr
# CHECK: encoding: [0x20,0xee,0x61]
xsr.excvaddr a2

# Instruction format RRR
# CHECK-INST: xsr a2, excvaddr
# CHECK: encoding: [0x20,0xee,0x61]
xsr a2, 238

# Instruction format RRR
# CHECK-INST: xsr a2, depc
# CHECK: encoding: [0x20,0xc0,0x61]
xsr a2, depc

# Instruction format RRR
# CHECK-INST: xsr a2, depc
# CHECK: encoding: [0x20,0xc0,0x61]
xsr.depc a2

# Instruction format RRR
# CHECK-INST: xsr a2, depc
# CHECK: encoding: [0x20,0xc0,0x61]
xsr a2, 192
