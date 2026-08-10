# RUN: llvm-mc %s -triple=mips-unknown-linux-gnu -show-encoding \
# RUN:   -mcpu=mips32r6 -mattr=+micromips,+ginv | FileCheck %s

  ginvi $4           # CHECK: ginvi $4         # encoding: [0x00,0x04,0x61,0x7c]
  ginvt $4, 2        # CHECK: ginvt $4, 2      # encoding: [0x00,0x04,0x75,0x7c]
