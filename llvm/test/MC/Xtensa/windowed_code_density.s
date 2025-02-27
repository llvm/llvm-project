# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+windowed,+density \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4

# Instruction format RRRN
# CHECK-INST: retw.n
# CHECK: encoding: [0x1d,0xf0]
retw.n

# Instruction format RRRN
# CHECK-INST: retw.n
# CHECK: encoding: [0x1d,0xf0]
_retw.n
