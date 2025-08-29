# RUN: llvm-mc -triple=hexagon -mv73 -filetype=obj %s | llvm-readobj -r - | FileCheck %s

# This checks various combinations of relocation addends.  Several cases below
# had been incorrect.

{
  call a
}
#CHECK: R_HEX_B22_PCREL a 0x0

## Expect .Lb+4
{
  if (p0) jump ##.Lb
}
{
  p0 = !cmp.gt(r2, #-1)
  p0 = sfclass(r2, #0xe)
  if (!p0.new) jump:t c
}
#CHECK: R_HEX_B32_PCREL_X c 0x8
#CHECK: R_HEX_B15_PCREL_X c 0xC

{
  if (!p0) jump d
  if (p0) jump d
}
#CHECK: R_HEX_B32_PCREL_X d 0x0
#CHECK: R_HEX_B15_PCREL_X d 0x4
#CHECK: R_HEX_B32_PCREL_X d 0x8
#CHECK: R_HEX_B15_PCREL_X d 0xC
{
  if (!p0) jump e
  jump .Lb
}
#CHECK: R_HEX_B32_PCREL_X e 0x0
#CHECK: R_HEX_B15_PCREL_X e 0x4
.Lb:

{
r0 = add(pc, ##foo@PCREL)
if (!p0) jump f
}
#CHECK: R_HEX_B32_PCREL_X foo 0x0
#CHECK: R_HEX_6_PCREL_X foo 0x4
#CHECK: R_HEX_B32_PCREL_X f 0x8
#CHECK: R_HEX_B15_PCREL_X f 0xC

{
r0 = add(pc, ##.Lx@PCREL)
if (!p0) jump __hexagon_sqrtf
}
.Lx:
#CHECK: R_HEX_B32_PCREL_X .text
#CHECK: R_HEX_6_PCREL_X .text
#CHECK: R_HEX_B32_PCREL_X __hexagon_sqrtf 0x8
#CHECK: R_HEX_B15_PCREL_X __hexagon_sqrtf 0xC
