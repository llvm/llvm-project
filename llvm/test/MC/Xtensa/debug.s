# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+debug,+density \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4
LBL0:

# Instruction format RRR
# CHECK-INST: break 1, 1
# CHECK: encoding: [0x10,0x41,0x00]
break 1, 1

# Instruction format RRRN
# CHECK-INST: break.n 1
# CHECK: encoding: [0x2d,0xf1]
break.n 1

# Instruction format RRR
# CHECK-INST: lddr32.p a3
# CHECK: encoding: [0xe0,0x73,0x00]
lddr32.p a3

# Instruction format RRR
# CHECK-INST: sddr32.p a3
# CHECK: encoding: [0xf0,0x73,0x00]
sddr32.p a3

#Instruction format RRR
#CHECK-INST: xsr a2, icount
#CHECK: encoding: [0x20,0xec,0x61]
xsr a2,icount

#Instruction format RRR
#CHECK-INST: xsr a2, icount
#CHECK: encoding: [0x20,0xec,0x61]
xsr.icount a2

#Instruction format RRR
#CHECK-INST: xsr a2, icount
#CHECK: encoding: [0x20,0xec,0x61]
xsr a2, 236

#Instruction format RRR
#CHECK-INST: xsr a2, icountlevel
#CHECK: encoding: [0x20,0xed,0x61]
xsr a2,icountlevel

#Instruction format RRR
#CHECK-INST: xsr a2, icountlevel
#CHECK: encoding: [0x20,0xed,0x61]
xsr.icountlevel a2

#Instruction format RRR
#CHECK-INST: xsr a2, icountlevel
#CHECK: encoding: [0x20,0xed,0x61]
xsr a2, 237

#Instruction format RRR
#CHECK-INST: xsr a2, ibreaka0
#CHECK: encoding: [0x20,0x80,0x61]
xsr a2,ibreaka0

#Instruction format RRR
#CHECK-INST: xsr a2, ibreaka0
#CHECK: encoding: [0x20,0x80,0x61]
xsr.ibreaka0 a2

#Instruction format RRR
#CHECK-INST: xsr a2, ibreaka0
#CHECK: encoding: [0x20,0x80,0x61]
xsr a2, 128

#Instruction format RRR
#CHECK-INST: xsr a2, ibreaka1
#CHECK: encoding: [0x20,0x81,0x61]
xsr a2,ibreaka1

#Instruction format RRR
#CHECK-INST: xsr a2, ibreaka1
#CHECK: encoding: [0x20,0x81,0x61]
xsr.ibreaka1 a2

#Instruction format RRR
#CHECK-INST: xsr a2, ibreaka1
#CHECK: encoding: [0x20,0x81,0x61]
xsr a2, 129

#Instruction format RRR
#CHECK-INST: xsr a2, dbreaka0
#CHECK: encoding: [0x20,0x90,0x61]
xsr a2,dbreaka0

#Instruction format RRR
#CHECK-INST: xsr a2, dbreaka0
#CHECK: encoding: [0x20,0x90,0x61]
xsr.dbreaka0 a2

#Instruction format RRR
#CHECK-INST: xsr a2, dbreaka0
#CHECK: encoding: [0x20,0x90,0x61]
xsr a2, 144

#Instruction format RRR
#CHECK-INST: xsr a2, dbreaka1
#CHECK: encoding: [0x20,0x91,0x61]
xsr a2,dbreaka1

#Instruction format RRR
#CHECK-INST: xsr a2, dbreaka1
#CHECK: encoding: [0x20,0x91,0x61]
xsr.dbreaka1 a2

#Instruction format RRR
#CHECK-INST: xsr a2, dbreaka1
#CHECK: encoding: [0x20,0x91,0x61]
xsr a2, 145

#Instruction format RRR
#CHECK-INST: xsr a2, dbreakc0
#CHECK: encoding: [0x20,0xa0,0x61]
xsr a2,dbreakc0

#Instruction format RRR
#CHECK-INST: xsr a2, dbreakc0
#CHECK: encoding: [0x20,0xa0,0x61]
xsr.dbreakc0 a2

#Instruction format RRR
#CHECK-INST: xsr a2, dbreakc0
#CHECK: encoding: [0x20,0xa0,0x61]
xsr a2, 160

#Instruction format RRR
#CHECK-INST: xsr a2, dbreakc1
#CHECK: encoding: [0x20,0xa1,0x61]
xsr a2,dbreakc1

#Instruction format RRR
#CHECK-INST: xsr a2, dbreakc1
#CHECK: encoding: [0x20,0xa1,0x61]
xsr.dbreakc1 a2

#Instruction format RRR
#CHECK-INST: xsr a2, dbreakc1
#CHECK: encoding: [0x20,0xa1,0x61]
xsr a2, 161

#Instruction format RRR
#CHECK-INST: xsr a2, ibreakenable
#CHECK: encoding: [0x20,0x60,0x61]
xsr a2,ibreakenable

#Instruction format RRR
#CHECK-INST: xsr a2, ibreakenable
#CHECK: encoding: [0x20,0x60,0x61]
xsr.ibreakenable a2

#Instruction format RRR
#CHECK-INST: xsr a2, ibreakenable
#CHECK: encoding: [0x20,0x60,0x61]
xsr a2, 96

#Instruction format RRR
#CHECK-INST: rsr a2, debugcause
#CHECK: encoding: [0x20,0xe9,0x03]
rsr a2,debugcause

#Instruction format RRR
#CHECK-INST: rsr a2, debugcause
#CHECK: encoding: [0x20,0xe9,0x03]
rsr.debugcause a2

#Instruction format RRR
#CHECK-INST: rsr a2, debugcause
#CHECK: encoding: [0x20,0xe9,0x03]
rsr a2, 233

#Instruction format RRR
#CHECK-INST: xsr a2, ddr
#CHECK: encoding: [0x20,0x68,0x61]
xsr a2,ddr

#Instruction format RRR
#CHECK-INST: xsr a2, ddr
#CHECK: encoding: [0x20,0x68,0x61]
xsr.ddr a2

#Instruction format RRR
#CHECK-INST: xsr a2, ddr
#CHECK: encoding: [0x20,0x68,0x61]
xsr a2, 104
