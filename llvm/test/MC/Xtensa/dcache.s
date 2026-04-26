# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+dcache \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4

# Instruction format RSR
# CHECK-INST: xsr a3, memctl
# CHECK: # encoding: [0x30,0x61,0x61]
xsr a3, memctl

# CHECK-INST: xsr a3, memctl
# CHECK: # encoding: [0x30,0x61,0x61]
xsr.memctl a3

# CHECK-INST: xsr a3, memctl
# CHECK: # encoding: [0x30,0x61,0x61]
xsr a3, 97
