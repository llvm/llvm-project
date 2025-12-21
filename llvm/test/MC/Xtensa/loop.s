# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+loop \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4

# Instruction format BRI8
# CHECK-INST: loop a3, LBL
# CHECK: encoding: [0x76,0x83,A]
loop a3, LBL

# Instruction format BRI8
# CHECK-INST: loopnez a3, LBL
# CHECK: encoding: [0x76,0x93,A]
loopnez a3, LBL

# Instruction format BRI8
# CHECK-INST: loopgtz a3, LBL
# CHECK: encoding: [0x76,0xa3,A]
loopgtz a3, LBL

# CHECK-INST: xsr a3, lbeg
# CHECK: # encoding: [0x30,0x00,0x61]
xsr a3, lbeg

# CHECK-INST: xsr a3, lend
# CHECK: # encoding: [0x30,0x01,0x61]
xsr a3, lend

# CHECK-INST: xsr a3, lcount
# CHECK: # encoding: [0x30,0x02,0x61]
xsr a3, lcount

.fill 200

LBL:
