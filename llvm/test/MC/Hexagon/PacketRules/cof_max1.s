# RUN: not llvm-mc -triple=hexagon -filetype=asm %s 2>&1 | FileCheck %s

{ jumpr r0
  jumpr r0 }
# CHECK: 3:3: error: Instruction may not be in a packet with other branches

{ jump unknown
  if (p0) jump unknown }
# CHECK: 7:3: error: Instruction may not be the first branch in packet

