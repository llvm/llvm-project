# RUN: llvm-mc -filetype=obj -triple=hexagon %s | llvm-objdump -r -d - | FileCheck %s
{
  call ##foo
  memw(##a) = r0
}
#CHECK: { 	immext(#0)
#CHECK: :  R_HEX_B32_PCREL_X	foo
#CHECK: call
#CHECK: R_HEX_B22_PCREL_X	foo
#CHECK: immext(#0)
#CHECK: R_HEX_32_6_X	a
#CHECK: memw(##0) = r0 }
#CHECK: R_HEX_16_X	a


