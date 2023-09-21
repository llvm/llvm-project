# RUN: not llvm-mc -triple=hexagon -filetype=asm %s 2>&1 | FileCheck %s

{ r0=sub(#1,r0)
  r1=sub(#1, r0)
  r2=memw(r0)
  dczeroa(r0) }
# CHECK: 5:3: note: Instruction was restricted from being in slot 1
# CHECK: 6:3: note: Instruction can only be combined with an ALU instruction in slot 1
# CHECK: 6:15: error: invalid instruction packet: slot error
