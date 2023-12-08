# RUN: llvm-mc %s -triple=xtensa -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s


.align	4
LBL0:

# Instruction format RRR
# CHECK-INST: dsync
# CHECK: encoding: [0x30,0x20,0x00]
dsync

# Instruction format RRR
# CHECK-INST: esync
# CHECK: encoding: [0x20,0x20,0x00]
esync

# Instruction format RRR
# CHECK-INST: isync
# CHECK: encoding: [0x00,0x20,0x00]
isync

# Instruction format RRR
# CHECK-INST: nop
# CHECK: encoding: [0xf0,0x20,0x00]
nop

# Instruction format RSR
# CHECK-INST: rsr a8, sar
# CHECK: encoding: [0x80,0x03,0x03]
rsr a8, sar

# CHECK-INST: rsr a8, sar
# CHECK: encoding: [0x80,0x03,0x03]
rsr.sar a8

# CHECK-INST: rsr a8, sar
# CHECK: encoding: [0x80,0x03,0x03]
rsr a8, 3

# Instruction format RRR
# CHECK-INST: rsync
# CHECK: encoding: [0x10,0x20,0x00]
rsync

# Instruction format RSR
# CHECK-INST: wsr a8, sar
# CHECK: encoding: [0x80,0x03,0x13]
wsr a8, sar

# CHECK-INST: wsr a8, sar
# CHECK: encoding: [0x80,0x03,0x13]
wsr.sar a8

# CHECK-INST: wsr a8, sar
# CHECK: encoding: [0x80,0x03,0x13]
wsr a8, 3

# Instruction format RRR
# CHECK-INST: xsr a8, sar
# CHECK: encoding: [0x80,0x03,0x61]
xsr a8, sar

# CHECK-INST: xsr a8, sar
# CHECK: encoding: [0x80,0x03,0x61]
xsr.sar a8

# CHECK-INST: xsr a8, sar
# CHECK: encoding: [0x80,0x03,0x61]
xsr a8, 3
