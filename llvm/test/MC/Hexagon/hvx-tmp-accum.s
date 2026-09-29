# RUN: not llvm-mc -triple=hexagon -mhvx -filetype=asm %s 2>%t; FileCheck --implicit-check-not="error:" %s <%t
{
    v0.tmp = vmem(r0+#0)
    v0 += vrmpyub(v1, r1)
}
# CHECK: error: register `V0.tmp' is accumulated in this packet

{
    v1.tmp = vmem(r0+#0)
    v1.w += vrmpy(v1.b,v2.b)
}
# CHECK: error: register `V1.tmp' is accumulated in this packet
