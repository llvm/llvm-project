# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+clamps \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4

# Instruction format RRR
# CHECK-INST: clamps a3, a4, 7
# CHECK: encoding: [0x00,0x34,0x33]
clamps a3, a4, 7
