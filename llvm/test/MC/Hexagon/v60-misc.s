# RUN: llvm-mc -arch=hexagon -mcpu=hexagonv60 -mhvx -filetype=obj %s | llvm-objdump --arch=hexagon --mcpu=hexagonv60 --mattr=+hvx -d - | FileCheck %s

.L0:

# CHECK: 5c00c000 { if (p0) jump:nt
if (p0) jump .L0

# CHECK: 5cffe1fe { if (!p1) jump:nt
if (!p1) jump .L0

# CHECK: 5340c200 { if (p2) jumpr:nt
if (p2) jumpr r0

# CHECK: 5361c300 { if (!p3) jumpr:nt
if (!p3) jumpr r1

# CHECK: 1c2eceee { v14 = vxor(v14,v14) }
v14 = #0

# CHECK: 1c9edea0 { v1:0.w = vsub(v31:30.w,v31:30.w) }
v1:0 = #0

# CHECK: 1f42c3e0 { v1:0 = vcombine(v3,v2) }
v1:0 = v3:2

# CHECK: 1f90cf00 { q0 = vcmp.eq(v15.b,v16.b) }
q0 = vcmp.eq(v15.ub, v16.ub)

# CHECK: 1c92f101 { q1 &= vcmp.eq(v17.b,v18.b) }
q1 &= vcmp.eq(v17.ub, v18.ub)

# CHECK: 1c94f342 { q2 |= vcmp.eq(v19.b,v20.b) }
q2 |= vcmp.eq(v19.ub, v20.ub)

# CHECK: 1c96f583 { q3 ^= vcmp.eq(v21.b,v22.b) }
q3 ^= vcmp.eq(v21.ub, v22.ub)

# CHECK: 1f81c004 { q0 = vcmp.eq(v0.h,v1.h) }
q0 = vcmp.eq(v0.uh, v1.uh)

# CHECK: 1c83e205 { q1 &= vcmp.eq(v2.h,v3.h) }
q1 &= vcmp.eq(v2.uh, v3.uh)

# CHECK: 1c85e446 { q2 |= vcmp.eq(v4.h,v5.h) }
q2 |= vcmp.eq(v4.uh, v5.uh)

# CHECK: 1c87e687 { q3 ^= vcmp.eq(v6.h,v7.h) }
q3 ^= vcmp.eq(v6.uh, v7.uh)

# CHECK: 1f89c808 { q0 = vcmp.eq(v8.w,v9.w) }
q0 = vcmp.eq(v8.uw, v9.uw)

# CHECK: 1c8aea09 { q1 &= vcmp.eq(v10.w,v10.w) }
q1 &= vcmp.eq(v10.uw, v10.uw)

# CHECK: 1c8ceb4a { q2 |= vcmp.eq(v11.w,v12.w) }
q2 |= vcmp.eq(v11.uw, v12.uw)

# CHECK: 1c8eed8b { q3 ^= vcmp.eq(v13.w,v14.w) }
q3 ^= vcmp.eq(v13.uw, v14.uw)

# CHECK: 2800c00f { v15 = vmem(r0+#0) }
v15 = vmem(r0)

# CHECK: 2841c010 { v16 = vmem(r1+#0):nt }
v16 = vmem(r1):nt

# CHECK: 2822c011 { vmem(r2+#0) = v17 }
vmem(r2) = v17

# CHECK: 2863c012 { vmem(r3+#0):nt = v18 }
vmem(r3):nt = v18

# CHECK: 2884c013 { if (q0) vmem(r4+#0) = v19 }
if (q0) vmem(r4) = v19

# CHECK: 2885c834 { if (!q1) vmem(r5+#0) = v20 }
if (!q1) vmem(r5) = v20

# CHECK: 28c6d015 { if (q2) vmem(r6+#0):nt = v21 }
if (q2) vmem(r6):nt = v21

# CHECK: 28c7d836 { if (!q3) vmem(r7+#0):nt = v22 }
if (!q3) vmem(r7):nt = v22

# CHECK: 28a8c017 { if (p0) vmem(r8+#0) = v23 }
if (p0) vmem(r8) = v23

# CHECK: 28a9c838 { if (!p1) vmem(r9+#0) = v24 }
if (!p1) vmem(r9) = v24

# CHECK: 28ead019 { if (p2) vmem(r10+#0):nt = v25 }
if (p2) vmem(r10):nt = v25

# CHECK: 28ebd83a { if (!p3) vmem(r11+#0):nt = v26 }
if (!p3) vmem(r11):nt = v26

# CHECK: 282cc022 vmem(r12+#0) = v27.new
{
  v27 = vxor(v28, v29)
  vmem(r12) = v27.new
}

# CHECK: 286dc022 vmem(r13+#0):nt = v30.new
{
  v30 = vxor(v31, v0)
  vmem(r13):nt = v30.new
}

# CHECK: 280ec0e1 { v1 = vmemu(r14+#0) }
v1 = vmemu(r14)

# CHECK: 282fc0e2 { vmemu(r15+#0) = v2 }
vmemu(r15) = v2

# CHECK: 28b0c0c3 { if (p0) vmemu(r16+#0) = v3 }
if (p0) vmemu(r16) = v3

# CHECK: 28b1c8e4 { if (!p1) vmemu(r17+#0) = v4 }
if (!p1) vmemu(r17) = v4

