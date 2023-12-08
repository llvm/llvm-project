# RUN: not llvm-mc %s -triple=mips-unknown-linux-gnu -show-encoding \
# RUN:     -mcpu=mips32r6 -mattr=+ginv 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -triple=mips64-unknown-linux-gnu -show-encoding \
# RUN:     -mcpu=mips64r6 -mattr=+ginv 2>%t1
# RUN: FileCheck %s < %t1

  .set noginv
  ginvi $4, 2  # CHECK: instruction requires a CPU feature not currently enabled
