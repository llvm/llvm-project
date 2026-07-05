# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+div32 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4

# Instruction format RRR
# CHECK-INST: quos a3, a4, a5
# CHECK: encoding: [0x50,0x34,0xd2]
quos a3, a4, a5

# Instruction format RRR
# CHECK-INST: quou a3, a4, a5
# CHECK: encoding: [0x50,0x34,0xc2]
quou a3, a4, a5

# Instruction format RRR
# CHECK-INST: rems a3, a4, a5
# CHECK: encoding:  [0x50,0x34,0xf2]
rems a3, a4, a5

# Instruction format RRR
# CHECK-INST: remu a3, a4, a5
# CHECK: encoding:  [0x50,0x34,0xe2]
remu a3, a4, a5
