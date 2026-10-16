# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+sext \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4

# Instruction format RRR
# CHECK-INST: sext a3, a4, 7
# CHECK: encoding: [0x00,0x34,0x23]
sext a3, a4, 7
