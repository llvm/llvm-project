# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+nsa \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4

# Instruction format RRR
# CHECK-INST: nsa a3, a4
# CHECK: encoding: [0x30,0xe4,0x40]
nsa a3, a4

# Instruction format RRR
# CHECK-INST: nsau a3, a4
# CHECK: encoding: [0x30,0xf4,0x40]
nsau a3, a4
