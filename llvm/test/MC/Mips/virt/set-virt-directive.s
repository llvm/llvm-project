# RUN: llvm-mc %s -show-encoding -triple=mips-unknown-linux-gnu \
# RUN:   -mcpu=mips32r5 | FileCheck %s
# RUN: llvm-mc %s -show-encoding -triple=mips64-unknown-linux-gnu \
# RUN:   -mcpu=mips64r5 | FileCheck %s

  .set virt
  hypcall  # CHECK: hypcall # encoding: [0x42,0x00,0x00,0x28]
