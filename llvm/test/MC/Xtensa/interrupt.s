# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+interrupt \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4
LBL0:

# Instruction format RRR
# CHECK-INST: rsil a2, 1
# CHECK: encoding: [0x20,0x61,0x00]
rsil a2, 1

# Instruction format RRR
# CHECK-INST: waiti 1
# CHECK: encoding: [0x00,0x71,0x00]
waiti 1

#Instruction format RRR
#CHECK-INST: rsr a2, interrupt
#CHECK: encoding: [0x20,0xe2,0x03]
rsr a2, interrupt

#Instruction format RRR
#CHECK-INST: rsr a2, interrupt
#CHECK: encoding: [0x20,0xe2,0x03]
rsr.interrupt a2

#Instruction format RRR
#CHECK-INST: rsr a2, interrupt
#CHECK: encoding: [0x20,0xe2,0x03]
rsr a2, 226

#Instruction format RRR
#CHECK-INST: wsr a2, intclear
#CHECK: encoding: [0x20,0xe3,0x13]
wsr a2, intclear

#Instruction format RRR
#CHECK-INST: wsr a2, intclear
#CHECK: encoding: [0x20,0xe3,0x13]
wsr.intclear a2

#Instruction format RRR
#CHECK-INST: wsr a2, intclear
#CHECK: encoding: [0x20,0xe3,0x13]
wsr a2, 227

#Instruction format RRR
#CHECK-INST: xsr a2, intenable
#CHECK: encoding: [0x20,0xe4,0x61]
xsr a2, intenable

#Instruction format RRR
#CHECK-INST: xsr a2, intenable
#CHECK: encoding: [0x20,0xe4,0x61]
xsr.intenable a2

#Instruction format RRR
#CHECK-INST: xsr a2, intenable
#CHECK: encoding: [0x20,0xe4,0x61]
xsr a2, 228
