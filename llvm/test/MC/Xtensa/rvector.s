# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+rvector \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4

# Instruction format RSR
# CHECK-INST: xsr a3, vecbase
# CHECK: # encoding: [0x30,0xe7,0x61]
xsr a3, vecbase

# CHECK-INST: xsr a3, vecbase
# CHECK: # encoding: [0x30,0xe7,0x61]
xsr.vecbase a3

# CHECK-INST: xsr a3, vecbase
# CHECK: # encoding: [0x30,0xe7,0x61]
xsr a3, 231
