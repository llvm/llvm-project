# RUN: llvm-mc -arch=hexagon -mcpu=hexagonv62 -filetype=obj %s | llvm-objdump -arch=hexagon -mcpu=hexagonv62 -d - | FileCheck %s

# verify compound is split into single instructions if needed
{
  p0=cmp.eq(R1:0,R3:2)
  if (!p0.new) jump:nt ltmp
  r0=r1 ; jump ltmp
}

# CHECK: 5c204800 { if (!p0.new) jump:nt
# CHECK: d2804200   p0 = cmp.eq(r1:0,r3:2)
# CHECK: 58004000   jump
# CHECK: 7061c000   r0 = r1 }
