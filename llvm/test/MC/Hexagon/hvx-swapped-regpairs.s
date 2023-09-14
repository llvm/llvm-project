# RUN: llvm-mc -filetype=obj -triple=hexagon -mcpu=hexagonv67 -mhvx %s | llvm-objdump --no-print-imm-hex -d --mcpu=hexagonv67 --mattr=+hvx - | FileCheck %s
# RUN: not llvm-mc -triple=hexagon -mcpu=hexagonv65 -mhvx -filetype=asm %s 2>%t; FileCheck --check-prefix=CHECK-V65 --implicit-check-not="error:" %s <%t

v1:0.w = vadd(v0.h, v1.h) // Normal
# CHECK: 1ca1c080

v0:1.w = vadd(v0.h, v1.h) // Swapped
# CHECK-NEXT: 1ca1c081
# CHECK-V65: error: register pair `WR0' is not permitted for this architecture

## Swapped use:
v1:0.w = vtmpy(v0:1.h,r0.b)
# CHECK-NEXT: 19a0c180
# CHECK-V65: error: register pair `WR0' is not permitted for this architecture

## Swapped def
v0:1 = v3:2
# CHECK-NEXT: 1f42c3e1 { v0:1 = vcombine(v3,v2) }
# CHECK-V65: error: register pair `WR0' is not permitted for this architecture

# Mapped instruction's swapped use:
v1:0 = v2:3
# CHECK-NEXT: v1:0 = vcombine(v2,v3)
## No error for v65, this is now permitted!

## .new producer from pair:
{
   v0:1 = vaddw(v0:1, v0:1)
   if (!p0) vmem(r0+#0)=v0.new
}
# CHECK-NEXT: v0:1.w = vadd(v0:1.w,v0:1.w)
# CHECK-NEXT: if (!p0) vmem(r0+#0) = v0.new
# CHECK-V65: error: register pair `WR0' is not permitted for this architecture

## Used .tmp, swapped use & def:
{
  v0.tmp = vmem(r0 + #3)
  v2:3 = vaddw(v0:1, v0:1)
}
# CHECK-NEXT: 1c6141c3 { v2:3.w = vadd(v0:1.w,v0:1.w)
# CHECK-NEXT:            v0.tmp = vmem(r0+#3) }
# CHECK-V65: error: register pair `WR0' is not permitted for this architecture
# CHECK-V65: error: register pair `WR1' is not permitted for this architecture
