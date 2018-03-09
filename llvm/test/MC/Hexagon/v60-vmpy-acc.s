#RUN: llvm-mc -triple=hexagon -mcpu=hexagonv60 -filetype=obj -mhvx %s | \
#RUN: llvm-objdump -triple=hexagon -mcpu=hexagonv60 -mhvx -d - | \
#RUN: FileCheck %s

#CHECK: 1936ee37 { v23.w += vdmpy(v15:14.h,r22.uh,#1):sat }
v23.w += vdmpy(v15:14.h,r22.uh,#1):sat

#CHECK: 193bf90f { v15.w += vdmpy(v25.h,r27.uh):sat }
v15.w += vdmpy(v25.h,r27.uh):sat

#CHECK: 1902fcf0 { v17:16.h += vdmpy(v29:28.ub,r2.b) }
v17:16.h += vdmpy(v29:28.ub,r2.b)

#CHECK: 190cffd1 { v17.h += vdmpy(v31.ub,r12.b) }
v17.h += vdmpy(v31.ub,r12.b)

#CHECK: 1900f5ac { v12.w += vrmpy(v21.ub,r0.b) }
v12.w += vrmpy(v21.ub,r0.b)

#CHECK: 1905fb86 { v6.uw += vrmpy(v27.ub,r5.ub) }
v6.uw += vrmpy(v27.ub,r5.ub)

#CHECK: 191de570 { v16.w += vdmpy(v5.h,r29.b) }
v16.w += vdmpy(v5.h,r29.b)

#CHECK: 191de846 { v7:6.w += vtmpy(v9:8.h,r29.b) }
v7:6.w += vtmpy(v9:8.h,r29.b)

#CHECK: 190bfa22 { v3:2.h += vtmpy(v27:26.ub,r11.b) }
v3:2.h += vtmpy(v27:26.ub,r11.b)

#CHECK: 1915e408 { v9:8.h += vtmpy(v5:4.b,r21.b) }
v9:8.h += vtmpy(v5:4.b,r21.b)

#CHECK: 1987f71e { v31:30.uh += vmpy(v23.ub,r7.ub) }
v31:30.uh += vmpy(v23.ub,r7.ub)

#CHECK: 1969ff47 { v7.w += vasl(v31.w,r9) }
v7.w += vasl(v31.w,r9)

#CHECK: 196de3b0 { v16.w += vasr(v3.w,r13) }
v16.w += vasr(v3.w,r13)

#CHECK: 1977fe0a { v11:10.uw += vdsad(v31:30.uh,r23.uh) }
v11:10.uw += vdsad(v31:30.uh,r23.uh)

#CHECK: 196eee36 { v22.h += vmpyi(v14.h,r14.b) }
v22.h += vmpyi(v14.h,r14.b)

#CHECK: 1931faac { v13:12.h += vmpy(v26.ub,r17.b) }
v13:12.h += vmpy(v26.ub,r17.b)

#CHECK: 193cfc94 { v21:20.w += vdmpy(v29:28.h,r28.b) }
v21:20.w += vdmpy(v29:28.h,r28.b)

#CHECK: 1934fc62 { v2.w += vdmpy(v28.h,r20.h):sat }
v2.w += vdmpy(v28.h,r20.h):sat

#CHECK: 1925fe5f { v31.w += vdmpy(v31:30.h,r5.h):sat }
v31.w += vdmpy(v31:30.h,r5.h):sat

#CHECK: 194efe36 { v23:22.uw += vmpy(v30.uh,r14.uh) }
v23:22.uw += vmpy(v30.uh,r14.uh)

#CHECK: 1948e306 { v7:6.w += vmpy(v3.h,r8.h):sat }
v7:6.w += vmpy(v3.h,r8.h):sat

#CHECK: 192af2f8 { v25:24.w += vmpa(v19:18.h,r10.b) }
v25:24.w += vmpa(v19:18.h,r10.b)

#CHECK: 1926e4da { v27:26.h += vmpa(v5:4.ub,r6.b) }
v27:26.h += vmpa(v5:4.ub,r6.b)

#CHECK: 194ff078 { v24.w += vmpyi(v16.w,r15.h) }
v24.w += vmpyi(v16.w,r15.h)

#CHECK: 1946e247 { v7.w += vmpyi(v2.w,r6.b) }
v7.w += vmpyi(v2.w,r6.b)

#CHECK: 1c3fead5 { v21.w += vmpyo(v10.w,v31.h):<<1:sat:shift }
v21.w += vmpyo(v10.w,v31.h):<<1:sat:shift

#CHECK: 1c30e1fa { v26.w += vmpyo(v1.w,v16.h):<<1:rnd:sat:shift }
v26.w += vmpyo(v1.w,v16.h):<<1:rnd:sat:shift

#CHECK: 1c34f690 { v16.h += vmpyi(v22.h,v20.h) }
v16.h += vmpyi(v22.h,v20.h)

#CHECK: 1c34f4b5 { v21.w += vmpyie(v20.w,v20.uh) }
v21.w += vmpyie(v20.w,v20.uh)

#CHECK: 1c54f804 { v4.w += vmpyie(v24.w,v20.h) }
v4.w += vmpyie(v24.w,v20.h)

#CHECK: 1c1ff6f4 { v21:20.w += vmpy(v22.h,v31.h) }
v21:20.w += vmpy(v22.h,v31.h)

#CHECK: 1c31f026 { v7:6.w += vmpy(v16.h,v17.uh) }
v7:6.w += vmpy(v16.h,v17.uh)

#CHECK: 1c12fb98 { v25:24.h += vmpy(v27.b,v18.b) }
v25:24.h += vmpy(v27.b,v18.b)

#CHECK: 1c17fcc0 { v1:0.h += vmpy(v28.ub,v23.b) }
v1:0.h += vmpy(v28.ub,v23.b)

#CHECK: 1c16f26f { v15.w += vdmpy(v18.h,v22.h):sat }
v15.w += vdmpy(v18.h,v22.h):sat

#CHECK: 1c0bea3a { v26.w += vrmpy(v10.b,v11.b) }
v26.w += vrmpy(v10.b,v11.b)

#CHECK: 1c15eb47 { v7.w += vrmpy(v11.ub,v21.b) }
v7.w += vrmpy(v11.ub,v21.b)

#CHECK: 1c26e40e { v15:14.uw += vmpy(v4.uh,v6.uh) }
v15:14.uw += vmpy(v4.uh,v6.uh)

#CHECK: 1c0df9a8 { v9:8.uh += vmpy(v25.ub,v13.ub) }
v9:8.uh += vmpy(v25.ub,v13.ub)

#CHECK: 1c0afc15 { v21.uw += vrmpy(v28.ub,v10.ub) }
v21.uw += vrmpy(v28.ub,v10.ub)
