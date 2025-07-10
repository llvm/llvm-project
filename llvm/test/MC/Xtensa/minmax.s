# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+minmax \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4

# Instruction format RRR
# CHECK-INST: max a3, a4, a5
# CHECK: encoding: [0x50,0x34,0x53]
max a3, a4, a5

# Instruction format RRR
# CHECK-INST: maxu a3, a4, a5
# CHECK: encoding: [0x50,0x34,0x73]
maxu a3, a4, a5

# Instruction format RRR
# CHECK-INST: min a3, a4, a5
# CHECK: encoding: [0x50,0x34,0x43]
min a3, a4, a5

# Instruction format RRR
# CHECK-INST: minu a3, a4, a5
# CHECK: encoding: [0x50,0x34,0x63]
minu a3, a4, a5
