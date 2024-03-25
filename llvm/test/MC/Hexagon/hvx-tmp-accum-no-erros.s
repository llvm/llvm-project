# RUN: llvm-mc -triple=hexagon -mattr=+hvxv68 -filetype=obj %s | llvm-objdump --no-print-imm-hex --mattr=+hvxv68 -d - | FileCheck %s

# packet w/accum with register different from one loaded to
{
    v1.tmp = vmem(r0+#0)
    v0.w += vrmpy(v1.b,v2.b)
}

# CHECK: { v0.w += vrmpy(v1.b,v2.b)
# CHECK-NEXT:  v1.tmp = vmem(r0+#0) }

# packet w/accum and store or other non-def register use
{
    v1.tmp = vmem(r0+#0)
    v0 += vrmpyub(v1, v3)
    vmem(r0) = v0
}

# CHECK: { v0.uw += vrmpy(v1.ub,v3.ub)
# CHECK-NEXT:  v1.tmp = vmem(r0+#0)
# CHECK-NEXT:  vmem(r0+#0) = v0 }

# packet w/non-accum and otherwise-legal register def/use
{
    v0.tmp =vmem(r2+#0)
    Q3 = vcmp.eq(v0.w, v5.w)
}

# CHECK: { q3 = vcmp.eq(v0.w,v5.w)
# CHECK-NEXT: v0.tmp = vmem(r2+#0) }

# scalar "accums" unaffected by this change.
{
    r0 += add(r1, r2)
}

# CHECK { r0 += add(r1,r2) }
