# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+coprocessor \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4
LBL0:

#Instruction format RRR
#CHECK-INST: xsr a2, cpenable
#CHECK: encoding: [0x20,0xe0,0x61]
xsr a2,cpenable

#Instruction format RRR
#CHECK-INST: xsr a2, cpenable
#CHECK: encoding: [0x20,0xe0,0x61]
xsr.cpenable a2

#Instruction format RRR
#CHECK-INST: xsr a2, cpenable
#CHECK: encoding: [0x20,0xe0,0x61]
xsr a2, 224
