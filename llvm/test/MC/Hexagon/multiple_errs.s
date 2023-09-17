# RUN: not llvm-mc -triple=hexagon -filetype=asm %s 2> %t; FileCheck %s < %t
#

{
  if (!p0) r0=r1;
  if (!p0) r0=r2;
  trap0(#15);
}
# CHECK: error: register `R0' modified more than once
# CHECK: error: Instruction is marked `isSolo' and cannot have other instructions in the same packet
