# RUN: not llvm-mc -arch=hexagon -filetype=asm -mcpu=hexagonv55 %s 2>%t; FileCheck %s < %t
#
{
  sp=asrh(r6)
  l2fetch(fp,r23:22)
  p2=r7
  p1=dfclass(r31:30,#6)
}
# CHECK: rror: Instruction can only
