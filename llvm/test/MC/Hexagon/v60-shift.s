#RUN: llvm-mc -triple=hexagon -mcpu=hexagonv60 -filetype=obj -mhvx %s | \
#RUN: llvm-objdump --triple=hexagon --mcpu=hexagonv60 --mattr=+hvx -d - | \
#RUN: FileCheck %s

#CHECK: 198fd829 { v9.uw = vlsr(v24.uw,{{ *}}r15) }
v9.uw=vlsr(v24.uw,r15)

#CHECK: 1999d645 { v5.uh = vlsr(v22.uh,{{ *}}r25) }
v5.uh=vlsr(v22.uh,r25)

#CHECK: 198cc303 { v3.h = vasl(v3.h,{{ *}}r12) }
v3.h=vasl(v3.h,r12)

#CHECK: 1965d7ac { v12.w = vasr(v23.w,{{ *}}r5) }
v12.w=vasr(v23.w,r5)

#CHECK: 197dddc3 { v3.h = vasr(v29.h,{{ *}}r29) }
v3.h=vasr(v29.h,r29)

#CHECK: 197adde8 { v8.w = vasl(v29.w,{{ *}}r26) }
v8.w=vasl(v29.w,r26)

#CHECK: 1977cc26 { v6 = vror(v12,{{ *}}r23) }
v6=vror(v12,r23)

#CHECK: 1e02cfad { v13.uw = vcl0(v15.uw) }
v13.uw=vcl0(v15.uw)

#CHECK: 1e02defb { v27.uh = vcl0(v30.uh) }
v27.uh=vcl0(v30.uh)

#CHECK: 1e03de90 { v16.w = vnormamt(v30.w) }
v16.w=vnormamt(v30.w)

#CHECK: 1e03d4a3 { v3.h = vnormamt(v20.h) }
v3.h=vnormamt(v20.h)

#CHECK: 1e02c2d8 { v24.h = vpopcount(v2.h) }
v24.h=vpopcount(v2.h)
