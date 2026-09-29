# RUN: llvm-mc -triple=hexagon -mcpu=hexagonv65 -mhvx -filetype=obj %s | \
# RUN:   llvm-objdump --no-print-imm-hex --mcpu=hexagonv65 --mattr=+hvx -d - | \
# RUN:   FileCheck %s

.L0:

# CHECK: 5c00c000 { if (p0) jump:nt
if (p0) jump .L0

# CHECK: 5cffe1fe { if (!p1) jump:nt
if (!p1) jump .L0

# CHECK: 5340c200 { if (p2) jumpr:nt
if (p2) jumpr r0

# CHECK: 5361c300 { if (!p3) jumpr:nt
if (!p3) jumpr r1

# CHECK: 1e64f1d7 { v23 = vlalign(v17,v4,#6) }
v23=vlalign(v17,v4,#6)

# CHECK: 1ec3c003 { q3 = and(q0,q3) }
q3=and(q0,q3)

# CHECK: 1e00ff3c { v29:28.w |= vunpacko(v31.h) }
v29:28.w|=vunpacko(v31.h)

# CHECK: 1e22e60e { v14 = valign(v6,v2,#0) }
v14=valign(v6,v2,#0)

# CHECK: 1baae196 { v23:22 = vdeal(v1,v21,r2) }
v23:22=vdeal(v1,v21,r2)

# CHECK: 1e00f80c { v13:12.h |= vunpacko(v24.b) }
v13:12.h|=vunpacko(v24.b)

# CHECK: 1b1ae609 { v9.b = vasr(v6.h,v3.h,r2):rnd:sat }
v9.b=vasr(v6.h,v3.h,r2):rnd:sat

# CHECK: 1ba8e77c { v29:28 = vshuff(v7,v21,r0) }
v29:28=vshuff(v7,v21,r0)

# CHECK: 1e43c107 { q3 = or(q1,q1) }
q3=or(q1,q1)

# CHECK: 1e03c20d { q1 = xor(q2,q0) }
q1=xor(q2,q0)

# CHECK: 1f8ecd19 { q1 = vcmp.gt(v13.w,v14.w) }
q1=vcmp.gt(v13.w,v14.w)

# CHECK: 1f9dce14 { q0 = vcmp.gt(v14.h,v29.h) }
q0=vcmp.gt(v14.h,v29.h)

# CHECK: 1e83c014 { q0 = and(q0,!q2) }
q0=and(q0,!q2)

# CHECK: 1e03c310 { q0 = or(q3,!q0) }
q0=or(q3,!q0)

# CHECK: 1e03c309 { q1 = not(q3) }
q1=not(q3)

# CHECK: 1e03c109 { q1 = not(q1) }
q1=not(q1)

# CHECK: 1f86d704 { q0 = vcmp.eq(v23.h,v6.h) }
q0=vcmp.eq(v23.h,v6.h)

# CHECK: 1f83d303 { q3 = vcmp.eq(v19.b,v3.b) }
q3=vcmp.eq(v19.b,v3.b)

# CHECK: 1f9fd110 { q0 = vcmp.gt(v17.b,v31.b) }
q0=vcmp.gt(v17.b,v31.b)

# CHECK: 1f99cd09 { q1 = vcmp.eq(v13.w,v25.w) }
q1=vcmp.eq(v13.w,v25.w)

# CHECK: 1a20d939 { if (!p1) v25 = v25 }
if (!p1) v25=v25

# CHECK: 1a00db33 { if (p1) v19 = v27 }
if (p1) v19=v27

# CHECK: 19fde252 { vdeal(v2,v18,r29) }
vdeal(v2,v18,r29)

# CHECK: 19eef43e { vshuff(v20,v30,r14) }
vshuff(v20,v30,r14)

# CHECK: 19bfd6cc { v13:12.uw = vrmpy(v23:22.ub,r31.ub,#0) }
v13:12.uw=vrmpy(v23:22.ub,r31.ub,#0)

# CHECK: 1946d4c4 { v5:4.uw = vrsad(v21:20.ub,r6.ub,#0) }
v5:4.uw=vrsad(v21:20.ub,r6.ub,#0)

# CHECK: 1941de94 { v21:20.w = vrmpy(v31:30.ub,r1.b,#0) }
v21:20.w=vrmpy(v31:30.ub,r1.b,#0)

# CHECK: 196ef8dc { v29:28.uw += vrmpy(v25:24.ub,r14.ub,#0) }
v29:28.uw+=vrmpy(v25:24.ub,r14.ub,#0)

# CHECK: 1944eaea { v11:10.uw += vrsad(v11:10.ub,r4.ub,#1) }
v11:10.uw+=vrsad(v11:10.ub,r4.ub,#1)

# CHECK: 1947fa9c { v29:28.w += vrmpy(v27:26.ub,r7.b,#0) }
v29:28.w+=vrmpy(v27:26.ub,r7.b,#0)

# CHECK: 19b4c0a5 { v5 = vand(q0,r20) }
v5=vand(q0,r20)

# CHECK: 19a3c02f { v15 = vsplat(r3) }
v15=vsplat(r3)

# CHECK: 197de377 { v23 |= vand(q3,r29) }
v23|=vand(q3,r29)

# CHECK: 196af580 { q0 |= vand(v21,r10) }
q0|=vand(v21,r10)

# CHECK: 197bf780 { q0 |= vand(v23,r27) }
q0|=vand(v23,r27)

# CHECK: 19b0c0a6 { v6 = vand(q0,r16) }
v6=vand(q0,r16)

# CHECK: 1f85d621 { q1 = vcmp.gt(v22.ub,v5.ub) }
q1=vcmp.gt(v22.ub,v5.ub)

# CHECK: 1f82dc25 { q1 = vcmp.gt(v28.uh,v2.uh) }
q1=vcmp.gt(v28.uh,v2.uh)

# CHECK: 1f80da29 { q1 = vcmp.gt(v26.uw,v0.uw) }
q1=vcmp.gt(v26.uw,v0.uw)

# CHECK: 1966e06a { v10 |= vand(q0,r6) }
v10|=vand(q0,r6)

# CHECK: 8204db68 { r9:8 -= rol(r5:4,#27) }
r9:8-=rol(r5:4,#27)

# CHECK: 8c01d47b { r27 = rol(r1,#20) }
r27=rol(r1,#20)

# CHECK: 8008ec6c { r13:12 = rol(r9:8,#44) }
r13:12=rol(r9:8,#44)

# CHECK: 19bcd349 { q1 = vand(v19,r28) }
q1=vand(v19,r28)

# CHECK: 19b1cb49 { q1 = vand(v11,r17) }
q1=vand(v11,r17)

# CHECK: 19b3c045 { q1 = vsetq(r19) }
q1=vsetq(r19)

# CHECK: 19aac044 { q0 = vsetq(r10) }
q0=vsetq(r10)

# CHECK: 19a0e034 { v20.w = vinsert(r0) }
v20.w=vinsert(r0)

# CHECK: 19b5e037 { v23.w = vinsert(r21) }
v23.w=vinsert(r21)

# CHECK: 19b1c026 { v6 = vsplat(r17) }
v6=vsplat(r17)

# CHECK: 8242e7ee { r15:14 |= rol(r3:2,#39) }
r15:14|=rol(r3:2,#39)

# CHECK: 829cc868 { r9:8 ^= rol(r29:28,#8) }
r9:8^=rol(r29:28,#8)

# CHECK: 8210cee0 { r1:0 += rol(r17:16,#14) }
r1:0+=rol(r17:16,#14)

# CHECK: 8256e17a { r27:26 &= rol(r23:22,#33) }
r27:26&=rol(r23:22,#33)

# CHECK: 8e49c97d { r29 &= rol(r9,#9) }
r29&=rol(r9,#9)

# CHECK: 8e49dde8 { r8 |= rol(r9,#29) }
r8|=rol(r9,#29)

# CHECK: 8e1ac76f { r15 -= rol(r26,#7) }
r15-=rol(r26,#7)

# CHECK: 8e06c3f0 { r16 += rol(r6,#3) }
r16+=rol(r6,#3)

# CHECK: 8e99c075 { r21 ^= rol(r25,#0) }
r21^=rol(r25,#0)

# CHECK: 9213db2e { r14 = vextract(v27,r19) }
r14=vextract(v27,r19)

# CHECK: a6a0cc00 { l2gclean(r13:12) }
l2gclean(r13:12)

# CHECK: a666c000 { l2unlocka(r6) }
l2unlocka(r6)

# CHECK: a0e8e000 { p0 = l2locka(r8) }
p0=l2locka(r8)

# CHECK: a6c0c400 { l2gcleaninv(r5:4) }
l2gcleaninv(r5:4)

# CHECK: a820c800 { l2gunlock }
l2gunlock

# CHECK: a820d800 { l2gcleaninv }
l2gcleaninv

# CHECK: a820d000 { l2gclean }
l2gclean

# CHECK: 1ea6fa00 { v1:0 = vswap(q0,v26,v6) }
v1:0=vswap(q0,v26,v6)

# CHECK: eaa8da5c { r29:28,p2 = vacsh(r9:8,r27:26) }
r29:28,p2=vacsh(r9:8,r27:26)

# CHECK: 1eeef124 { v4 = vmux(q1,v17,v14) }
v4=vmux(q1,v17,v14)

# CHECK: 1bb2e928 { v8.b = vlut32(v9.b,v22.b,r2) }
v8.b=vlut32(v9.b,v22.b,r2)

# CHECK: 1b13e0fa { v27:26.h |= vlut16(v0.b,v2.h,r3) }
v27:26.h|=vlut16(v0.b,v2.h,r3)

# CHECK: 1b8ad836 { v22 = vlalign(v24,v17,r2) }
v22=vlalign(v24,v17,r2)

# CHECK: 1b41dd14 { v20 = valign(v29,v8,r1) }
v20=valign(v29,v8,r1)

# CHECK: 1a5ed41e { if (!p0) v31:30 = vcombine(v20,v30) }
if (!p0) v31:30=vcombine(v20,v30)

# CHECK: 1a7cc216 { if (p0) v23:22 = vcombine(v2,v28) }
if (p0) v23:22=vcombine(v2,v28)

# CHECK: 1bf9d389 { v9.h = vasr(v19.w,v31.w,r1):rnd:sat }
v9.h=vasr(v19.w,v31.w,r1):rnd:sat

# CHECK: 1bc9cb56 { v22.h = vasr(v11.w,v25.w,r1) }
v22.h=vasr(v11.w,v25.w,r1)

# CHECK: 1b08d2c5 { v5.ub = vasr(v18.h,v1.h,r0):sat }
v5.ub=vasr(v18.h,v1.h,r0):sat

# CHECK: 1befd0f4 { v20.ub = vasr(v16.h,v29.h,r7):rnd:sat }
v20.ub=vasr(v16.h,v29.h,r7):rnd:sat

# CHECK: 1b86f0d2 { v19:18.h = vlut16(v16.b,v16.h,r6) }
v19:18.h=vlut16(v16.b,v16.h,r6)

# CHECK: 1b9cf6bf { v31.b |= vlut32(v22.b,v19.b,r4) }
v31.b|=vlut32(v22.b,v19.b,r4)

# CHECK: 1b76d6ab { v11.uh = vasr(v22.w,v14.w,r6):sat }
v11.uh=vasr(v22.w,v14.w,r6):sat

# CHECK: 1b14c06f { v15.h = vasr(v0.w,v2.w,r4):sat }
v15.h=vasr(v0.w,v2.w,r4):sat

# CHECK: 1c2eceee { v14 = vxor(v14,v14) }
v14=#0

# CHECK: 1c4eceee { v14.w = vsub(v14.w,v14.w) }
v14.w=vsub(v14.w, v14.w)

# CHECK: 19e8eb2a { vshuff(v11,v10,r8) }
vtrans2x2(v11, v10, r8)

# CHECK: 537ad100 { if (!p1) jumpr:t
if (!p1) jumpr:t r26

# CHECK: 5354d300 { if (p3) jumpr:t
if (p3) jumpr:t r20

# CHECK: 5c20d100   if (!p1) jump:t
if (!p1) jump:t l579

# CHECK: 5c00d100   if (p1) jump:t
if (p1) jump:t l1143

# CHECK: 1baeeb8a { v11:10 = vdeal(v11,v21,r6) }
v11:10=vdeal(v11,v21,r6)

# CHECK: 1e00e080 { vhist }
vhist

# CHECK: 1e42e080 { vhist(q1) }
vhist(q1)

# CHECK: 1f42c3e0 { v1:0 = vcombine(v3,v2) }
v1:0=v3:2

# CHECK: 1f90cf00 { q0 = vcmp.eq(v15.b,v16.b) }
q0=vcmp.eq(v15.ub, v16.ub)

# CHECK: 1c92f101 { q1 &= vcmp.eq(v17.b,v18.b) }
q1&=vcmp.eq(v17.ub, v18.ub)

# CHECK: 1c94f342 { q2 |= vcmp.eq(v19.b,v20.b) }
q2|=vcmp.eq(v19.ub, v20.ub)

# CHECK: 1c96f583 { q3 ^= vcmp.eq(v21.b,v22.b) }
q3^=vcmp.eq(v21.ub, v22.ub)

# CHECK: 1f81c004 { q0 = vcmp.eq(v0.h,v1.h) }
q0=vcmp.eq(v0.uh, v1.uh)

# CHECK: 1c83e205 { q1 &= vcmp.eq(v2.h,v3.h) }
q1&=vcmp.eq(v2.uh, v3.uh)

# CHECK: 1c85e446 { q2 |= vcmp.eq(v4.h,v5.h) }
q2|=vcmp.eq(v4.uh, v5.uh)

# CHECK: 1c87e687 { q3 ^= vcmp.eq(v6.h,v7.h) }
q3^=vcmp.eq(v6.uh, v7.uh)

# CHECK: 1f89c808 { q0 = vcmp.eq(v8.w,v9.w) }
q0=vcmp.eq(v8.uw, v9.uw)

# CHECK: 1c8aea09 { q1 &= vcmp.eq(v10.w,v10.w) }
q1&=vcmp.eq(v10.uw, v10.uw)

# CHECK: 1c8ceb46 { q2 |= vcmp.eq(v11.h,v12.h) }
q2|=vcmp.eq(v11.uh, v12.uh)

# CHECK: 1c8eed8b { q3 ^= vcmp.eq(v13.w,v14.w) }
q3^=vcmp.eq(v13.uw, v14.uw)
