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

# Instruction format RRR
# CHECK-INST: rsync
# CHECK: encoding: [0x10,0x20,0x00]
rsync
