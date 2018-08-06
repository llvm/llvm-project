# RUN: llvm-mc %s -triple=mips-unknown-linux-gnu -show-encoding \
# RUN:   -mcpu=mips32r6 -mattr=+ginv | FileCheck %s
# RUN: llvm-mc %s -triple=mips64-unknown-linux-gnu -show-encoding \
# RUN:   -mcpu=mips64r6 -mattr=+ginv | FileCheck %s

  .set ginv
  ginvi $4  # CHECK: ginvi $4  # encoding: [0x7c,0x80,0x00,0x3d]
