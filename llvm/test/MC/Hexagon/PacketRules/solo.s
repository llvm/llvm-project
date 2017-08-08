# RUN: not llvm-mc -arch=hexagon -filetype=asm %s 2>%t; FileCheck %s <%t

{ brkpt
  r0 = r0 }
# CHECK: 3:3: error: Instruction is marked `isSolo' and cannot have other instructions in the same packet
