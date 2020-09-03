# RUN: not llvm-mc -arch=hexagon -mcpu=hexagonv67 -mhvx -filetype=asm %s 2>%t; FileCheck  --implicit-check-not="error:" %s <%t

{
  v1:0 = #0
  v0:1 = #0
}
# CHECK: error: register `V1' modified more than once

## Unused .tmp:
{
  v1.tmp = vmem(r0 + #3)
  v0:1 = vaddw(v17:16, v17:16)
}

# CHECK: warning: register `V1' used with `.tmp' but not used in the same packet
