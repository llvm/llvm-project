# RUN: llvm-mc -triple=hexagon -mv65 -mhvx -filetype=asm < %s | FileCheck %s

# Test that a slot error is not reported for a packet with a load and a
# vscatter.

# CHECK: vscatter(r0,m0,v0.h).h = v1
{
  v1=vmem(r1+#0)
  vscatter(r0,m0,v0.h).h=v1
}
# CHECK: vscatter(r2,m0,v1:0.w).h += v2
{
  v1=vmem(r3+#0)
  vscatter(r2,m0,v1:0.w).h+=v2
}
# CHECK: vmem(r4+#0):scatter_release
{
  v1=vmem(r5+#0)
  vmem(r4+#0):scatter_release
}
# CHECK: vmem(r4+#0):scatter_release
{
  v1=vmem(r5+#0)
  vmem(r4+#0):scatter_release
}
