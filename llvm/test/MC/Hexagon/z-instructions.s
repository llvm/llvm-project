# RUN: llvm-mc -arch=hexagon -mcpu=hexagonv66 -mhvx -filetype=obj %s | llvm-objdump --no-print-imm-hex --mcpu=hexagonv66 --mattr=+hvx -d - | FileCheck --implicit-check-not='{' %s

# CHECK:      2d00c000 { z = vmem(r0++#0) }
z = vmem(r0++#0)

# CHECK-NEXT: 2c00c000 { z = vmem(r0+#0) }
z = vmem(r0+#0)

# CHECK-NEXT: 2d00c001 { z = vmem(r0++m0) }
z = vmem(r0++m0)

# CHECK-NEXT: { v3:0.w += vrmpyz(v13.b,r3.b++)
# CHECK-NEXT:   v13.tmp = vmem(r2++#1)
# CHECK-NEXT:   z = vmem(r3+#0) }
{ v13.tmp = vmem(r2++#1)
  v3:0.w += vrmpyz(v13.b,r3.b++)
  z = vmem(r3+#0) }
