# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+windowed \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4
LBL0:

# Instruction format BRI12
# CHECK-INST: entry a3, 128
# CHECK: encoding: [0x36,0x03,0x01]
entry a3, 128

# Instruction format RRR
# CHECK-INST: movsp a3, a4
# CHECK: encoding: [0x30,0x14,0x00]
movsp a3, a4

# Instruction format CALL
# CHECK-INST: call4	LBL0
# CHECK: encoding: [0bAA010101,A,A]
call4 LBL0

# Instruction format CALL
# CHECK-INST: call8 LBL0
# CHECK: encoding: [0bAA100101,A,A]
call8 LBL0

# Instruction format CALL
# CHECK-INST: call12 LBL0
# CHECK: encoding: [0bAA110101,A,A]
call12 LBL0

# Instruction format CALLX
# CHECK-INST: callx4 a3
# CHECK: encoding: [0xd0,0x03,0x00]
callx4 a3

# Instruction format CALLX
# CHECK-INST: callx8 a3
# CHECK: encoding: [0xe0,0x03,0x00]
callx8 a3

# Instruction format CALLX
# CHECK-INST: callx12 a3
# CHECK: encoding: [0xf0,0x03,0x00]
callx12 a3

# Instruction format CALLX
# CHECK-INST: retw
# CHECK: encoding: [0x90,0x00,0x00]
retw

# Instruction format CALLX
# CHECK-INST: retw
# CHECK: encoding: [0x90,0x00,0x00]
_retw

# Instruction format RRR
# CHECK-INST: rotw 2
# CHECK: encoding: [0x20,0x80,0x40]
rotw 2

# Instruction format RRI4
# CHECK-INST: l32e a3, a4, -12
# CHECK: encoding: [0x30,0xd4,0x09]
l32e a3, a4, -12

# Instruction format RRI4
# CHECK-INST: s32e a3, a4, -12
# CHECK: encoding: [0x30,0xd4,0x49]
s32e a3, a4, -12

# Instruction format RRR
# CHECK-INST: rfwo
# CHECK: encoding: [0x00,0x34,0x00]
rfwo

# Instruction format RRR
# CHECK-INST: rfwu
# CHECK: encoding: [0x00,0x35,0x00]
rfwu

# Instruction format RSR
# CHECK-INST: xsr a3, windowbase
# CHECK: # encoding: [0x30,0x48,0x61]
xsr a3, windowbase

# CHECK-INST: xsr a3, windowbase
# CHECK: # encoding: [0x30,0x48,0x61]
xsr.windowbase a3

# CHECK-INST: xsr a3, windowbase
# CHECK: # encoding: [0x30,0x48,0x61]
xsr a3, 72

# CHECK-INST: xsr a3, windowstart
# CHECK: # encoding: [0x30,0x49,0x61]
xsr a3, windowstart

# CHECK-INST: xsr a3, windowstart
# CHECK: # encoding: [0x30,0x49,0x61]
xsr.windowstart a3

# CHECK-INST: xsr a3, windowstart
# CHECK: # encoding: [0x30,0x49,0x61]
xsr a3, 73
