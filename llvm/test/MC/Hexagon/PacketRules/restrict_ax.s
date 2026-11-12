# RUN: not llvm-mc -triple=hexagon -filetype=asm %s 2>%t; FileCheck %s <%t

{ r0=memw_locked(r0)
  r1=sfadd(r0,r0) }
# CHECK: 3:3: error: Instruction can only be in a packet with ALU or non-FPU XTYPE instructions
