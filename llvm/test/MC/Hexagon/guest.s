# RUN: llvm-mc -arch=hexagon -mcpu=hexagonv62 -filetype=obj %s | llvm-objdump -d - | FileCheck %s

	r0=gpmucnt4
 # CHECK: { r0 = gpmucnt4 }
	r0=gpmucnt5
 # CHECK: { r0 = gpmucnt5 }
	r0=gpmucnt6
 # CHECK: { r0 = gpmucnt6 }
	r0=gpmucnt7
 # CHECK: { r0 = gpmucnt7 }
	r0=gpcyclelo
 # CHECK: { r0 = gpcyclelo }
	r0=gpcyclehi
 # CHECK: { r0 = gpcyclehi }
	r0=gpmucnt0
 # CHECK: { r0 = gpmucnt0 }
	r0=gpmucnt1
 # CHECK: { r0 = gpmucnt1 }
	r0=gpmucnt2
 # CHECK: { r0 = gpmucnt2 }
	r0=gpmucnt3
 # CHECK: { r0 = gpmucnt3 }
	r0=gelr
 # CHECK: { r0 = gelr }
	r0=gsr
 # CHECK: { r0 = gsr }
	r0=gosp
 # CHECK: { r0 = gosp }
	r0=gbadva
 # CHECK: { r0 = gbadva }

	r1:0=g1:0
 # CHECK: { r1:0 = g1:0 }
	r1:0=g3:2
 # CHECK: { r1:0 = g3:2 }
	r1:0=g17:16
 # CHECK: { r1:0 = g17:16 }
	r1:0=g19:18
 # CHECK: { r1:0 = g19:18 }
	r1:0=g25:24
 # CHECK: { r1:0 = g25:24 }
	r1:0=g27:26
 # CHECK: { r1:0 = g27:26 }
	r1:0=g29:28
 # CHECK: { r1:0 = g29:28 }

{
        if (!p1) callr r26
        r17=g0
        if (!p3) r26=or(r15,r9)
        memb(r11+#-478)=r17.new
}
# CHECK:  { r17 = gelr
# CHECK:    if (!p1) callr r26
# CHECK:    if (!p3) r26 = or(r15,r9)
# CHECK:    memb(r11+#-478) = r17.new }

{
        if (!p1) callr r26
        r17=gpmucnt2
        if (!p3) r26=or(r15,r9)
        memb(r11+#-478)=r17.new
}
# CHECK:  { r17 = gpmucnt2
# CHECK:    if (!p1) callr r26
# CHECK:    if (!p3) r26 = or(r15,r9)
# CHECK:    memb(r11+#-478) = r17.new }
