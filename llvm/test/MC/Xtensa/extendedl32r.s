# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+extendedl32r \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4

# Instruction format RSR
# CHECK-INST: xsr a3, litbase
# CHECK: # encoding: [0x30,0x05,0x61]
xsr a3, litbase

# CHECK-INST: xsr a3, litbase
# CHECK: # encoding: [0x30,0x05,0x61]
xsr.litbase a3

# CHECK-INST: xsr a3, litbase
# CHECK: # encoding: [0x30,0x05,0x61]
xsr a3, 5
