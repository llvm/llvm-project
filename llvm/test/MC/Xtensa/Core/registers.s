# RUN: llvm-mc %s -triple=xtensa -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s


.align	4
LBL0:

#############################################################
## Check special registers parsing
#############################################################

# CHECK-INST: xsr a8, sar
# CHECK: encoding: [0x80,0x03,0x61]
xsr a8, 3
