# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+prid \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4
LBL0:

#Instruction format RRR
#CHECK-INST: rsr a2, prid
#CHECK: encoding: [0x20,0xeb,0x03]
rsr a2,prid

#Instruction format RRR
#CHECK-INST: rsr a2, prid
#CHECK: encoding: [0x20,0xeb,0x03]
rsr.prid a2

#Instruction format RRR
#CHECK-INST: rsr a2, prid
#CHECK: encoding: [0x20,0xeb,0x03]
rsr a2, 235
