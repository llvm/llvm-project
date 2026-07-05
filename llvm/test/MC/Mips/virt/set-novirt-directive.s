# RUN: not llvm-mc %s -triple=mips-unknown-linux-gnu -show-encoding \
# RUN:     -mcpu=mips32r5 -mattr=+virt 2>%t1
# RUN: FileCheck %s < %t1
# RUN: not llvm-mc %s -triple=mips64-unknown-linux-gnu -show-encoding \
# RUN:     -mcpu=mips64r5 -mattr=+virt 2>%t1
# RUN: FileCheck %s < %t1

  .set novirt
  hypcall  # CHECK: instruction requires a CPU feature not currently enabled
