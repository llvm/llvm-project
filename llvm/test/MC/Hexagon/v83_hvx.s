# RUN: llvm-mc -triple=hexagon -mv83 -mhvx -filetype=obj %s | \
# RUN:   llvm-objdump --mcpu=hexagonv83 --mattr=+hvx -d - | FileCheck %s

# Test the new V83 HVX instructions. CHECK lines verify the encoding bytes.

# V6_vabsdiffuw
v0.uw = vabsdiff(v1.uw,v2.uw)
# CHECK: 1aa2e140

# V6_vadduhhsat
v0.uh = vadd(v1.h,v2.h):sat
# CHECK: 1a82e1c0

# V6_vadduwwsat_dv
v1:0.uw = vadd(v3:2.w,v5:4.w):sat
# CHECK: 1a04e2c0

# V6_vmaxuw
v0.uw = vmax(v1.uw,v2.uw)
# CHECK: 1aa2e100

# V6_vminuw
v0.uw = vmin(v1.uw,v2.uw)
# CHECK: 1aa2e120

# V6_vnavguh
v0.h = vnavg(v1.uh,v2.uh)
# CHECK: 1aa2e160

# V6_vnavguw
v0.w = vnavg(v1.uw,v2.uw)
# CHECK: 1aa2e180

# V6_vsubuhhsat
v0.uh = vsub(v1.h,v2.h):sat
# CHECK: 1a02e1a0

# V6_vsubuwwsat_dv
v1:0.uw = vsub(v3:2.w,v5:4.w):sat
# CHECK: 1a04e2e0

# V6_vmpar_uwuhuw_si
v3:2.w = vmpa(v3:2.h,v1.uh,r0.w)
# CHECK: 1aa0e1a2

# V6_vmpar_wuhw_si
v3:2.uw = vmpa(v3:2.uh,v1.uh,r0.w)
# CHECK: 1aa0e1c2

# V6_vmpauhuh
v1.h = vmpa(v1.h,v1.uh,r1:0.uh)
# CHECK: 19e0e161

# V6_vmpsuhuh
v1.h = vmps(v1.h,v1.uh,r1:0.uh)
# CHECK: 19e0e181

# V6_vmpyewuh_rnd
v0.w = vmpye(v1.w,v2.uh):rnd
# CHECK: 1982c180

# V6_vmpyhuh
v0.h = vmpy(v1.h,v2.uh)
# CHECK: 1a82e100

# V6_vmpyhuh_rnd
v0.h = vmpy(v1.h,v2.uh):rnd
# CHECK: 1a82e140

# V6_vmpyowuh
v0.w = vmpyo(v1.w,v2.uh)
# CHECK: 1c42e1c0

# V6_vmpyowuh_acc
v1.w += vmpyo(v1.w,v2.uh)
# CHECK: 1c42e1e1

# V6_vmpyowuh_rnd
v0.w = vmpyo(v1.w,v2.uh):rnd
# CHECK: 1d02c180

# V6_vmpyowuh_rnd_acc
v1.w += vmpyo(v1.w,v2.uh):rnd
# CHECK: 1d02c1a1

# V6_vmpyrhuh
v0.h = vmpy(v1.h,r0.uh)
# CHECK: 1a80e120

# V6_vmpyrhuh_rnd
v0.h = vmpy(v1.h,r0.uh):rnd
# CHECK: 1a80e160

# V6_vmpyrhus
v1:0.w = vmpy(v1.h,r0.uh)
# CHECK: 1a80e180

# V6_vmpyrhus_acc
v3:2.w += vmpy(v1.h,r0.uh)
# CHECK: 1a80e1a2

# V6_vmpyruhuh
v0.uh = vmpy(v1.uh,r0.uh)
# CHECK: 19e0e1a0

# V6_vmpyruhuh_rnd
v0.uh = vmpy(v1.uh,r0.uh):rnd
# CHECK: 19e0e1e0

# V6_vmpyuhuh_rnd
v0.uh = vmpy(v1.uh,v2.uh):rnd
# CHECK: 19e2e1c0

# V6_vmpauhuhuh
v1.uh = vmpa(v1.uh,v1.uh,r1:0.uh)
# CHECK: 19e0e101
