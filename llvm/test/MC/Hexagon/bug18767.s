# RUN: not llvm-mc -arch=hexagon -filetype=asm %s 2>%t; FileCheck %s < %t

.L_:
{
  loop0(.L_,r2);
  r7=r1;
  r5=mpyi(r1,#64);
  r6=#0;
  nop;
}
# CHECK: rror: invalid instruction packet: out of slots
