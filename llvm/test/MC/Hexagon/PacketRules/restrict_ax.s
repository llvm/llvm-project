{ r0=memw_locked(r0)
  r1=-mpyi(r0,#0) }
# RUN: not llvm-mc -arch=hexagon -filetype=asm %s 2>%t; FileCheck %s --check-prefix=CHECK00 <%t
# CHECK00: 1:3: error: Instruction can only be in a packet with ALU or non-FPU XTYPE instructions
