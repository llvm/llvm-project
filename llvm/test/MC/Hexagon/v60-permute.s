#RUN: llvm-mc -triple=hexagon -mcpu=hexagonv60 -filetype=obj -mhvx %s | \
#RUN: llvm-objdump --triple=hexagon --mcpu=hexagonv60 --mattr=+hvx -d - | \
#RUN: FileCheck %s

#CHECK: 1fd2d5cf { v15.b = vpack(v21.h{{ *}},{{ *}}v18.h):sat }
v15.b=vpack(v21.h,v18.h):sat

#CHECK: 1fd7d7a2 { v2.ub = vpack(v23.h{{ *}},{{ *}}v23.h):sat }
v2.ub=vpack(v23.h,v23.h):sat

#CHECK: 1fc7d464 { v4.h = vpacke(v20.w{{ *}},{{ *}}v7.w) }
v4.h=vpacke(v20.w,v7.w)

#CHECK: 1fc2c75b { v27.b = vpacke(v7.h{{ *}},{{ *}}v2.h) }
v27.b=vpacke(v7.h,v2.h)

#CHECK: 1fc9c5ed { v13.uh = vpack(v5.w{{ *}},{{ *}}v9.w):sat }
v13.uh=vpack(v5.w,v9.w):sat

#CHECK: 1ff1d81f { v31.h = vpack(v24.w{{ *}},{{ *}}v17.w):sat }
v31.h=vpack(v24.w,v17.w):sat

#CHECK: 1fe6c435 { v21.b = vpacko(v4.h{{ *}},{{ *}}v6.h) }
v21.b=vpacko(v4.h,v6.h)

#CHECK: 1febc140 { v0.h = vpacko(v1.w{{ *}},{{ *}}v11.w) }
v0.h=vpacko(v1.w,v11.w)

#CHECK: 1e01d256 { v23:22.h = vunpack(v18.b) }
v23:22.h=vunpack(v18.b)

#CHECK: 1e01cc38 { v25:24.uw = vunpack(v12.uh) }
v25:24.uw=vunpack(v12.uh)

#CHECK: 1e01c61e { v31:30.uh = vunpack(v6.ub) }
v31:30.uh=vunpack(v6.ub)

#CHECK: 1e01d778 { v25:24.w = vunpack(v23.h) }
v25:24.w=vunpack(v23.h)

#CHECK: 1e00c0e0 { v0.b = vdeal(v0.b) }
v0.b=vdeal(v0.b)

#CHECK: 1e00d5c9 { v9.h = vdeal(v21.h) }
v9.h=vdeal(v21.h)

#CHECK: 1e02cb1c { v28.b = vshuff(v11.b) }
v28.b=vshuff(v11.b)

#CHECK: 1e01d8fe { v30.h = vshuff(v24.h) }
v30.h=vshuff(v24.h)
