#RUN: llvm-mc -triple=hexagon -mcpu=hexagonv60 -filetype=obj -mhvx %s | \
#RUN: llvm-objdump -triple=hexagon -mcpu=hexagonv60 -mhvx -d - | \
#RUN: FileCheck %s

#CHECK: 1ce2cbd7 { v23.w = vavg(v11.w,{{ *}}v2.w):rnd }
v23.w=vavg(v11.w,v2.w):rnd

#CHECK: 1cf4d323 { v3.h = vnavg(v19.h,{{ *}}v20.h) }
v3.h=vnavg(v19.h,v20.h)

#CHECK: 1cffce9a { v26.uh = vavg(v14.uh,{{ *}}v31.uh):rnd }
v26.uh=vavg(v14.uh,v31.uh):rnd

#CHECK: 1ce5cba1 { v1.h = vavg(v11.h,{{ *}}v5.h):rnd }
v1.h=vavg(v11.h,v5.h):rnd

#CHECK: 1cc0d012 { v18.ub = vabsdiff(v16.ub,{{ *}}v0.ub) }
v18.ub=vabsdiff(v16.ub,v0.ub)

#CHECK: 1cc2de29 { v9.uh = vabsdiff(v30.h,{{ *}}v2.h) }
v9.uh=vabsdiff(v30.h,v2.h)

#CHECK: 1ce9ca06 { v6.b = vnavg(v10.ub,{{ *}}v9.ub) }
v6.b=vnavg(v10.ub,v9.ub)

#CHECK: 1caacf90 { v17:16.w = vadd(v15.h,{{ *}}v10.h) }
v17:16.w=vadd(v15.h,v10.h)

#CHECK: 1cb4cabe { v31:30.h = vsub(v10.ub,{{ *}}v20.ub) }
v31:30.h=vsub(v10.ub,v20.ub)

#CHECK: 1cb8cada { v27:26.w = vsub(v10.uh,{{ *}}v24.uh) }
v27:26.w=vsub(v10.uh,v24.uh)

#CHECK: 1cbcdbe8 { v9:8.w = vsub(v27.h,{{ *}}v28.h) }
v9:8.w=vsub(v27.h,v28.h)

#CHECK: 1caeca00 { v1:0.h = vsub(v11:10.h,{{ *}}v15:14.h):sat }
v1:0.h=vsub(v11:10.h,v15:14.h):sat

#CHECK: 1ca8c43e { v31:30.w = vsub(v5:4.w,{{ *}}v9:8.w):sat }
v31:30.w=vsub(v5:4.w,v9:8.w):sat

#CHECK: 1cbad95c { v29:28.h = vadd(v25.ub,{{ *}}v26.ub) }
v29:28.h=vadd(v25.ub,v26.ub)

#CHECK: 1ca1dc64 { v5:4.w = vadd(v28.uh,{{ *}}v1.uh) }
v5:4.w=vadd(v28.uh,v1.uh)

#CHECK: 1c79c350 { v16.h = vsub(v3.h,{{ *}}v25.h):sat }
v16.h=vsub(v3.h,v25.h):sat

#CHECK: 1c7fd364 { v4.w = vsub(v19.w,{{ *}}v31.w):sat }
v4.w=vsub(v19.w,v31.w):sat

#CHECK: 1c67d816 { v22.ub = vsub(v24.ub,{{ *}}v7.ub):sat }
v22.ub=vsub(v24.ub,v7.ub):sat

#CHECK: 1c7ddc2f { v15.uh = vsub(v28.uh,{{ *}}v29.uh):sat }
v15.uh=vsub(v28.uh,v29.uh):sat

#CHECK: 1c5cc6d7 { v23.h = vsub(v6.h,{{ *}}v28.h) }
v23.h=vsub(v6.h,v28.h)

#CHECK: 1c54cae4 { v4.w = vsub(v10.w,{{ *}}v20.w) }
v4.w=vsub(v10.w,v20.w)

#CHECK: 1c4dc78b { v11.w = vadd(v7.w,{{ *}}v13.w):sat }
v11.w=vadd(v7.w,v13.w):sat

#CHECK: 1c48c7a4 { v4.b = vsub(v7.b,{{ *}}v8.b) }
v4.b=vsub(v7.b,v8.b)

#CHECK: 1cdec3b0 { v16.uh = vavg(v3.uh,{{ *}}v30.uh) }
v16.uh=vavg(v3.uh,v30.uh)

#CHECK: 1c76dc98 { v25:24.b = vadd(v29:28.b,{{ *}}v23:22.b) }
v25:24.b=vadd(v29:28.b,v23:22.b)

#CHECK: 1c7ad4a6 { v7:6.h = vadd(v21:20.h,{{ *}}v27:26.h) }
v7:6.h=vadd(v21:20.h,v27:26.h)

#CHECK: 1cc7c564 { v4.uw = vabsdiff(v5.w,{{ *}}v7.w) }
v4.uw=vabsdiff(v5.w,v7.w)

#CHECK: 1cd2cdc1 { v1.h = vavg(v13.h,{{ *}}v18.h) }
v1.h=vavg(v13.h,v18.h)

#CHECK: 1cd5d246 { v6.uh = vabsdiff(v18.uh,{{ *}}v21.uh) }
v6.uh=vabsdiff(v18.uh,v21.uh)

#CHECK: 1cdcd987 { v7.ub = vavg(v25.ub,{{ *}}v28.ub) }
v7.ub=vavg(v25.ub,v28.ub)

#CHECK: 1c92c6e4 { v5:4.uh = vsub(v7:6.uh,{{ *}}v19:18.uh):sat }
v5:4.uh=vsub(v7:6.uh,v19:18.uh):sat

#CHECK: 1c86dace { v15:14.ub = vsub(v27:26.ub,{{ *}}v7:6.ub):sat }
v15:14.ub=vsub(v27:26.ub,v7:6.ub):sat

#CHECK: 1cffc07c { v28.ub = vavg(v0.ub,{{ *}}v31.ub):rnd }
v28.ub=vavg(v0.ub,v31.ub):rnd

#CHECK: 1cf8d851 { v17.w = vnavg(v24.w,{{ *}}v24.w) }
v17.w=vnavg(v24.w,v24.w)

#CHECK: 1c70d2e6 { v7:6.ub = vadd(v19:18.ub,{{ *}}v17:16.ub):sat }
v7:6.ub=vadd(v19:18.ub,v17:16.ub):sat

#CHECK: 1c72dec6 { v7:6.w = vadd(v31:30.w,{{ *}}v19:18.w) }
v7:6.w=vadd(v31:30.w,v19:18.w)

#CHECK: 1c92d23e { v31:30.h = vadd(v19:18.h,{{ *}}v19:18.h):sat }
v31:30.h=vadd(v19:18.h,v19:18.h):sat

#CHECK: 1c94de1e { v31:30.uh = vadd(v31:30.uh,{{ *}}v21:20.uh):sat }
v31:30.uh=vadd(v31:30.uh,v21:20.uh):sat

#CHECK: 1c9ec07c { v29:28.b = vsub(v1:0.b,{{ *}}v31:30.b) }
v29:28.b=vsub(v1:0.b,v31:30.b)

#CHECK: 1c88da56 { v23:22.w = vadd(v27:26.w,{{ *}}v9:8.w):sat }
v23:22.w=vadd(v27:26.w,v9:8.w):sat

#CHECK: 1c9acab8 { v25:24.w = vsub(v11:10.w,{{ *}}v27:26.w) }
v25:24.w=vsub(v11:10.w,v27:26.w)

#CHECK: 1c82d282 { v3:2.h = vsub(v19:18.h,{{ *}}v3:2.h) }
v3:2.h=vsub(v19:18.h,v3:2.h)

#CHECK: 1c2bd9a6 { v6 = vand(v25,{{ *}}v11) }
v6=vand(v25,v11)

#CHECK: 1c43c22d { v13.ub = vadd(v2.ub,{{ *}}v3.ub):sat }
v13.ub=vadd(v2.ub,v3.ub):sat

#CHECK: 1c59d707 { v7.w = vadd(v23.w,{{ *}}v25.w) }
v7.w=vadd(v23.w,v25.w)

#CHECK: 1c3fc9e1 { v1 = vxor(v9,{{ *}}v31) }
v1=vxor(v9,v31)

#CHECK: 1c2acbdf { v31 = vor(v11,{{ *}}v10) }
v31=vor(v11,v10)

#CHECK: 1cdaccf6 { v22.w = vavg(v12.w,{{ *}}v26.w) }
v22.w=vavg(v12.w,v26.w)

#CHECK: 1c5ac767 { v7.h = vadd(v7.h,{{ *}}v26.h):sat }
v7.h=vadd(v7.h,v26.h):sat

#CHECK: 1c40d956 { v22.uh = vadd(v25.uh,{{ *}}v0.uh):sat }
v22.uh=vadd(v25.uh,v0.uh):sat

#CHECK: 1fbbd611 { v17.w = vasr(v22.w{{ *}},{{ *}}v27.w) }
v17.w=vasr(v22.w,v27.w)

#CHECK: 1fbad835 { v21.w = vlsr(v24.w{{ *}},{{ *}}v26.w) }
v21.w=vlsr(v24.w,v26.w)

#CHECK: 1f79cedc { v28.b = vround(v14.h{{ *}},{{ *}}v25.h):sat }
v28.b=vround(v14.h,v25.h):sat

#CHECK: 1f69c4e0 { v0.ub = vround(v4.h{{ *}},{{ *}}v9.h):sat }
v0.ub=vround(v4.h,v9.h):sat

#CHECK: 1f72c485 { v5.h = vround(v4.w{{ *}},{{ *}}v18.w):sat }
v5.h=vround(v4.w,v18.w):sat

#CHECK: 1f6bc8b1 { v17.uh = vround(v8.w{{ *}},{{ *}}v11.w):sat }
v17.uh=vround(v8.w,v11.w):sat

#CHECK: 1f71c25b { v27.ub = vsat(v2.h{{ *}},{{ *}}v17.h) }
v27.ub=vsat(v2.h,v17.h)

#CHECK: 1f66c560 { v0.h = vsat(v5.w{{ *}},{{ *}}v6.w) }
v0.h=vsat(v5.w,v6.w)

#CHECK: 1fb3d148 { v8.h = vlsr(v17.h{{ *}},{{ *}}v19.h) }
v8.h=vlsr(v17.h,v19.h)

#CHECK: 1fbec56e { v14.h = vasr(v5.h{{ *}},{{ *}}v30.h) }
v14.h=vasr(v5.h,v30.h)

#CHECK: 1fb2d2a2 { v2.h = vasl(v18.h{{ *}},{{ *}}v18.h) }
v2.h=vasl(v18.h,v18.h)

#CHECK: 1faccc95 { v21.w = vasl(v12.w{{ *}},{{ *}}v12.w) }
v21.w=vasl(v12.w,v12.w)

#CHECK: 1fb9c1e2 { v2.h = vadd(v1.h{{ *}},{{ *}}v25.h) }
v2.h=vadd(v1.h,v25.h)

#CHECK: 1fbbd5df { v31.b = vadd(v21.b{{ *}},{{ *}}v27.b) }
v31.b=vadd(v21.b,v27.b)

#CHECK: 1f25c578 { v24 = vrdelta(v5{{ *}},{{ *}}v5) }
v24=vrdelta(v5,v5)

#CHECK: 1f22c62a { v10 = vdelta(v6{{ *}},{{ *}}v2) }
v10=vdelta(v6,v2)

#CHECK: 1f20d102 { v2.w = vmax(v17.w{{ *}},{{ *}}v0.w) }
v2.w=vmax(v17.w,v0.w)

#CHECK: 1f1ed6fc { v28.h = vmax(v22.h{{ *}},{{ *}}v30.h) }
v28.h=vmax(v22.h,v30.h)

#CHECK: 1f0cc8d8 { v24.uh = vmax(v8.uh{{ *}},{{ *}}v12.uh) }
v24.uh=vmax(v8.uh,v12.uh)

#CHECK: 1f00c1b0 { v16.ub = vmax(v1.ub{{ *}},{{ *}}v0.ub) }
v16.ub=vmax(v1.ub,v0.ub)

#CHECK: 1f12d08e { v14.w = vmin(v16.w{{ *}},{{ *}}v18.w) }
v14.w=vmin(v16.w,v18.w)

#CHECK: 1f1ad466 { v6.h = vmin(v20.h{{ *}},{{ *}}v26.h) }
v6.h=vmin(v20.h,v26.h)

#CHECK: 1f13df5d { v29.uh = vmin(v31.uh{{ *}},{{ *}}v19.uh) }
v29.uh=vmin(v31.uh,v19.uh)

#CHECK: 1f09c226 { v6.ub = vmin(v2.ub{{ *}},{{ *}}v9.ub) }
v6.ub=vmin(v2.ub,v9.ub)

#CHECK: 1f41d34f { v15.b = vshuffo(v19.b{{ *}},{{ *}}v1.b) }
v15.b=vshuffo(v19.b,v1.b)

#CHECK: 1f5fc72e { v14.b = vshuffe(v7.b{{ *}},{{ *}}v31.b) }
v14.b=vshuffe(v7.b,v31.b)

#CHECK: 1f34d0f7 { v23.b = vdeale(v16.b{{ *}},{{ *}}v20.b) }
v23.b=vdeale(v16.b,v20.b)

#CHECK: 1f4bd6c4 { v5:4.b = vshuffoe(v22.b{{ *}},{{ *}}v11.b) }
v5:4.b=vshuffoe(v22.b,v11.b)

#CHECK: 1f5dcea2 { v3:2.h = vshuffoe(v14.h{{ *}},{{ *}}v29.h) }
v3:2.h=vshuffoe(v14.h,v29.h)

#CHECK: 1f4fd186 { v6.h = vshuffo(v17.h{{ *}},{{ *}}v15.h) }
v6.h=vshuffo(v17.h,v15.h)

#CHECK: 1f5bda79 { v25.h = vshuffe(v26.h{{ *}},{{ *}}v27.h) }
v25.h=vshuffe(v26.h,v27.h)

#CHECK: 1f41d1f2 { v19:18 = vcombine(v17{{ *}},{{ *}}v1) }
v19:18=vcombine(v17,v1)

#CHECK: 1e82f432 { if (!q2) v18.b -= v20.b }
if (!q2) v18.b-=v20.b

#CHECK: 1ec2fd13 { if (q3) v19.w -= v29.w }
if (q3) v19.w-=v29.w

#CHECK: 1e81fef9 { if (q2) v25.h -= v30.h }
if (q2) v25.h-=v30.h

#CHECK: 1e81e2d3 { if (q2) v19.b -= v2.b }
if (q2) v19.b-=v2.b

#CHECK: 1e41ecad { if (!q1) v13.w += v12.w }
if (!q1) v13.w+=v12.w

#CHECK: 1e41e789 { if (!q1) v9.h += v7.h }
if (!q1) v9.h+=v7.h

#CHECK: 1e81e967 { if (!q2) v7.b += v9.b }
if (!q2) v7.b+=v9.b

#CHECK: 1e41f04f { if (q1) v15.w += v16.w }
if (q1) v15.w+=v16.w

#CHECK: 1e01e838 { if (q0) v24.h += v8.h }
if (q0) v24.h+=v8.h

#CHECK: 1ec1f112 { if (q3) v18.b += v17.b }
if (q3) v18.b+=v17.b

#CHECK: 1e42f67b { if (!q1) v27.w -= v22.w }
if (!q1) v27.w-=v22.w

#CHECK: 1e82ea5b { if (!q2) v27.h -= v10.h }
if (!q2) v27.h-=v10.h

#CHECK: 1e00c586 { v6 = vnot(v5) }
v6=vnot(v5)

#CHECK: 1e00df70 { v16.w = vabs(v31.w):sat }
v16.w=vabs(v31.w):sat

#CHECK: 1e00d45f { v31.w = vabs(v20.w) }
v31.w=vabs(v20.w)

#CHECK: 1e00db2f { v15.h = vabs(v27.h):sat }
v15.h=vabs(v27.h):sat

#CHECK: 1e00d001 { v1.h = vabs(v16.h) }
v1.h=vabs(v16.h)

#CHECK: 1e02c832 { v19:18.uh = vzxt(v8.ub) }
v19:18.uh=vzxt(v8.ub)

#CHECK: 1e02c98a { v11:10.w = vsxt(v9.h) }
v11:10.w=vsxt(v9.h)

#CHECK: 1e02cf76 { v23:22.h = vsxt(v15.b) }
v23:22.h=vsxt(v15.b)

#CHECK: 1e02c258 { v25:24.uw = vzxt(v2.uh) }
v25:24.uw=vzxt(v2.uh)
