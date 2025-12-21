# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+miscsr \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4

# Instruction format RSR
# CHECK-INST: xsr a3, misc0
# CHECK: # encoding: [0x30,0xf4,0x61]
xsr a3, misc0

# CHECK-INST: xsr a3, misc0
# CHECK: # encoding: [0x30,0xf4,0x61]
xsr.misc0 a3

# CHECK-INST: xsr a3, misc0
# CHECK: # encoding: [0x30,0xf4,0x61]
xsr a3, 244

# Instruction format RSR
# CHECK-INST: xsr a3, misc1
# CHECK: # encoding: [0x30,0xf5,0x61]
xsr a3, misc1

# CHECK-INST: xsr a3, misc1
# CHECK: # encoding: [0x30,0xf5,0x61]
xsr.misc1 a3

# CHECK-INST: xsr a3, misc1
# CHECK: # encoding: [0x30,0xf5,0x61]
xsr a3, 245

# Instruction format RSR
# CHECK-INST: xsr a3, misc2
# CHECK: # encoding: [0x30,0xf6,0x61]
xsr a3, misc2

# CHECK-INST: xsr a3, misc2
# CHECK: # encoding: [0x30,0xf6,0x61]
xsr.misc2 a3

# CHECK-INST: xsr a3, misc2
# CHECK: # encoding: [0x30,0xf6,0x61]
xsr a3, 246

# Instruction format RSR
# CHECK-INST: xsr a3, misc3
# CHECK: # encoding: [0x30,0xf7,0x61]
xsr a3, misc3

# CHECK-INST: xsr a3, misc3
# CHECK: # encoding: [0x30,0xf7,0x61]
xsr.misc3 a3

# CHECK-INST: xsr a3, misc3
# CHECK: # encoding: [0x30,0xf7,0x61]
xsr a3, 247
