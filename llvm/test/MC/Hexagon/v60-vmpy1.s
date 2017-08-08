#RUN: llvm-mc -triple=hexagon -mcpu=hexagonv60 -filetype=obj %s | \
#RUN: llvm-objdump -triple=hexagon -mcpu=hexagonv60 -d - | \
#RUN: FileCheck %s

#CHECK: 1939c223 { v3.w = vdmpy(v3:2.h,{{ *}}r25.uh,{{ *}}#1):sat }
v3.w=vdmpy(v3:2.h,r25.uh,#1):sat

#CHECK: 1936de0d { v13.w = vdmpy(v30.h,{{ *}}r22.uh):sat }
v13.w=vdmpy(v30.h,r22.uh):sat

#CHECK: 1919ccea { v11:10.h = vdmpy(v13:12.ub,{{ *}}r25.b) }
v11:10.h=vdmpy(v13:12.ub,r25.b)

#CHECK: 1918ced6 { v22.h = vdmpy(v14.ub,{{ *}}r24.b) }
v22.h=vdmpy(v14.ub,r24.b)

#CHECK: 1911deba { v27:26.uw = vdsad(v31:30.uh,{{ *}}r17.uh) }
v27:26.uw=vdsad(v31:30.uh,r17.uh)

#CHECK: 1908da97 { v23.w = vrmpy(v26.ub,{{ *}}r8.b) }
v23.w=vrmpy(v26.ub,r8.b)

#CHECK: 1915c974 { v20.uw = vrmpy(v9.ub,{{ *}}r21.ub) }
v20.uw=vrmpy(v9.ub,r21.ub)

#CHECK: 190dd446 { v6.w = vdmpy(v20.h,{{ *}}r13.b) }
v6.w=vdmpy(v20.h,r13.b)

#CHECK: 190ec030 { v17:16.h = vtmpy(v1:0.ub,{{ *}}r14.b) }
v17:16.h=vtmpy(v1:0.ub,r14.b)

#CHECK: 1918de1c { v29:28.h = vtmpy(v31:30.b,{{ *}}r24.b) }
v29:28.h=vtmpy(v31:30.b,r24.b)

#CHECK: 198dddf1 { v17.w = vmpyi(v29.w,{{ *}}r13.h) }
v17.w=vmpyi(v29.w,r13.h)

#CHECK: 19bccb13 { v19.w = vmpyi(v11.w,{{ *}}r28.b) }
v19.w=vmpyi(v11.w,r28.b)

#CHECK: 19c8cb0a { v11:10.uh = vmpy(v11.ub,{{ *}}r8.ub) }
v11:10.uh=vmpy(v11.ub,r8.ub)

#CHECK: 1973d012 { v18.h = vmpyi(v16.h,{{ *}}r19.b) }
v18.h=vmpyi(v16.h,r19.b)

#CHECK: 1922d1aa { v11:10.h = vmpy(v17.ub,{{ *}}r2.b) }
v11:10.h=vmpy(v17.ub,r2.b)

#CHECK: 1936ce9c { v29:28.w = vdmpy(v15:14.h,{{ *}}r22.b) }
v29:28.w=vdmpy(v15:14.h,r22.b)

#CHECK: 1925d86b { v11.w = vdmpy(v25:24.h,{{ *}}r5.h):sat }
v11.w=vdmpy(v25:24.h,r5.h):sat

#CHECK: 1925c255 { v21.w = vdmpy(v2.h,{{ *}}r5.h):sat }
v21.w=vdmpy(v2.h,r5.h):sat

#CHECK: 1941d424 { v4.h = vmpy(v20.h,{{ *}}r1.h):<<1:sat }
v4.h=vmpy(v20.h,r1.h):<<1:sat

#CHECK: 1943cf0a { v11:10.w = vmpy(v15.h,{{ *}}r3.h) }
v11:10.w=vmpy(v15.h,r3.h)

#CHECK: 193ec2f0 { v17:16.w = vmpa(v3:2.h,{{ *}}r30.b) }
v17:16.w=vmpa(v3:2.h,r30.b)

#CHECK: 193ddcde { v31:30.h = vmpa(v29:28.ub,{{ *}}r29.b) }
v31:30.h=vmpa(v29:28.ub,r29.b)

#CHECK: 1946de76 { v23:22.uw = vmpy(v30.uh,{{ *}}r6.uh) }
v23:22.uw=vmpy(v30.uh,r6.uh)

#CHECK: 1945c945 { v5.h = vmpy(v9.h,{{ *}}r5.h):<<1:rnd:sat }
v5.h=vmpy(v9.h,r5.h):<<1:rnd:sat

#CHECK: 19b0c280 { v1:0.w = vtmpy(v3:2.h,{{ *}}r16.b) }
v1:0.w=vtmpy(v3:2.h,r16.b)

#CHECK: 1c34d937 { v23.h = vmpy(v25.h,{{ *}}v20.h):<<1:rnd:sat }
v23.h=vmpy(v25.h,v20.h):<<1:rnd:sat

#CHECK: 1c36c90a { v11:10.uw = vmpy(v9.uh,{{ *}}v22.uh) }
v11:10.uw=vmpy(v9.uh,v22.uh)

#CHECK: 1c09c3ec { v13:12.w = vmpy(v3.h,{{ *}}v9.h) }
v13:12.w=vmpy(v3.h,v9.h)

#CHECK: 1c0dd1d8 { v25:24.h = vmpy(v17.ub,{{ *}}v13.b) }
v25:24.h=vmpy(v17.ub,v13.b)

#CHECK: 1c0dc0a4 { v5:4.uh = vmpy(v0.ub,{{ *}}v13.ub) }
v5:4.uh=vmpy(v0.ub,v13.ub)

#CHECK: 1c14df84 { v5:4.h = vmpy(v31.b,{{ *}}v20.b) }
v5:4.h=vmpy(v31.b,v20.b)

#CHECK: 1c16d77c { v28.w = vdmpy(v23.h,{{ *}}v22.h):sat }
v28.w=vdmpy(v23.h,v22.h):sat

#CHECK: 1c08d84f { v15.w = vrmpy(v24.ub,{{ *}}v8.b) }
v15.w=vrmpy(v24.ub,v8.b)

#CHECK: 1c06da29 { v9.w = vrmpy(v26.b,{{ *}}v6.b) }
v9.w=vrmpy(v26.b,v6.b)

#CHECK: 1c1ac805 { v5.uw = vrmpy(v8.ub,{{ *}}v26.ub) }
v5.uw=vrmpy(v8.ub,v26.ub)

#CHECK: 1c39d089 { v9.h = vmpyi(v16.h,{{ *}}v25.h) }
v9.h=vmpyi(v16.h,v25.h)

#CHECK: 1c3ecc64 { v5:4.h = vmpa(v13:12.ub,{{ *}}v31:30.b) }
v5:4.h=vmpa(v13:12.ub,v31:30.b)

#CHECK: 1c21ce54 { v21:20.w = vmpy(v14.h,{{ *}}v1.uh) }
v21:20.w=vmpy(v14.h,v1.uh)

#CHECK: 1cf2c6f0 { v17:16.h = vmpa(v7:6.ub,{{ *}}v19:18.ub) }
v17:16.h=vmpa(v7:6.ub,v19:18.ub)

#CHECK: 1fcdc82b { v11.w = vmpyio(v8.w{{ *}},{{ *}}v13.h) }
v11.w=vmpyio(v8.w,v13.h)

#CHECK: 1fdeda10 { v16.w = vmpyie(v26.w{{ *}},{{ *}}v30.uh) }
v16.w=vmpyie(v26.w,v30.uh)

#CHECK: 1ff2c2a6 { v6.w = vmpye(v2.w{{ *}},{{ *}}v18.uh) }
v6.w=vmpye(v2.w,v18.uh)

#CHECK: 1ff7cbfa { v26.w = vmpyo(v11.w{{ *}},{{ *}}v23.h):<<1:sat }
v26.w=vmpyo(v11.w,v23.h):<<1:sat

#CHECK: 1f5cd411 { v17.w = vmpyo(v20.w{{ *}},{{ *}}v28.h):<<1:rnd:sat }
v17.w=vmpyo(v20.w,v28.h):<<1:rnd:sat

#CHECK: 1f71cf1d { v29.w = vmpyieo(v15.h{{ *}},{{ *}}v17.h) }
v29.w=vmpyieo(v15.h,v17.h)
