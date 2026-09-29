# RUN: llvm-mc %s -triple=xtensa -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s


.align	4
LBL0:

# Instruction format CALL
# CHECK-INST:  call0   LBL0
# CHECK: encoding: [0bAA000101,A,A]
call0  LBL0

# Instruction format CALLX
# CHECK-INST:  callx0  a1
# CHECK: encoding: [0xc0,0x01,0x00]
callx0 a1

# Instruction format CALL
# CHECK-INST:  j       LBL0
# CHECK: encoding: [0bAA000110,A,A]
j LBL0

# Instruction format CALLX
# CHECK-INST:  jx      a2
# CHECK: encoding: [0xa0,0x02,0x00]
jx a2

# Instruction format CALLX
# CHECK-INST: ret
# CHECK: encoding: [0x80,0x00,0x00]
ret
