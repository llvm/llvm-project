# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+timers3 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4
LBL0:

#Instruction format RRR
#CHECK-INST: xsr a2, ccount
#CHECK: encoding: [0x20,0xea,0x61]
xsr a2,ccount

#Instruction format RRR
#CHECK-INST: xsr a2, ccount
#CHECK: encoding: [0x20,0xea,0x61]
xsr.ccount a2

#Instruction format RRR
#CHECK-INST: xsr a2, ccount
#CHECK: encoding: [0x20,0xea,0x61]
xsr a2, 234

#Instruction format RRR
#CHECK-INST: xsr a2, ccompare0
#CHECK: encoding: [0x20,0xf0,0x61]
xsr a2,ccompare0

#Instruction format RRR
#CHECK-INST: xsr a2, ccompare0
#CHECK: encoding: [0x20,0xf0,0x61]
xsr.ccompare0 a2

#Instruction format RRR
#CHECK-INST: xsr a2, ccompare0
#CHECK: encoding: [0x20,0xf0,0x61]
xsr a2, 240

#Instruction format RRR
#CHECK-INST: xsr a2, ccompare1
#CHECK: encoding: [0x20,0xf1,0x61]
xsr a2,ccompare1

#Instruction format RRR
#CHECK-INST: xsr a2, ccompare1
#CHECK: encoding: [0x20,0xf1,0x61]
xsr.ccompare1 a2

#Instruction format RRR
#CHECK-INST: xsr a2, ccompare1
#CHECK: encoding: [0x20,0xf1,0x61]
xsr a2, 241

#Instruction format RRR
#CHECK-INST: xsr a2, ccompare2
#CHECK: encoding: [0x20,0xf2,0x61]
xsr a2,ccompare2

#Instruction format RRR
#CHECK-INST: xsr a2, ccompare2
#CHECK: encoding: [0x20,0xf2,0x61]
xsr.ccompare2 a2

#Instruction format RRR
#CHECK-INST: xsr a2, ccompare2
#CHECK: encoding: [0x20,0xf2,0x61]
xsr a2, 242
