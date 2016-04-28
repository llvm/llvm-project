# RUN: llvm-mc -arch=hexagon -mcpu=hexagonv60 -mattr=+hvx -filetype=obj %s | llvm-objdump -arch=hexagon -mcpu=hexagonv60 -mattr=+hvx -d - | FileCheck %s

# CHECK: 1c2eceee { v14 = vxor(v14,{{ *}}v14) }
v14 = #0

# CHECK: 1c80c0a0 { v1:0.w = vsub(v1:0.w,v1:0.w) }
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

# CHECK: 1c8ceb46 { q2 |= vcmp.eq(v11.h,v12.h) }
q2 |= vcmp.eq(v11.uw, v12.uw)

# CHECK: 1c8eed8b { q3 ^= vcmp.eq(v13.w,v14.w) }
q3 ^= vcmp.eq(v13.uw, v14.uw)
