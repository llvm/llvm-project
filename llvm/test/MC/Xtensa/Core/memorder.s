# RUN: llvm-mc %s -triple=xtensa -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s


.align	4
LBL0:

# Instruction format RRR
# CHECK-INST: extw
# CHECK: encoding: [0xd0,0x20,0x00]
extw

# Instruction format RRR
# CHECK-INST: memw
# CHECK: encoding: [0xc0,0x20,0x00]
memw
