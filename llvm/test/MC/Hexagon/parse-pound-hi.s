# RUN: llvm-mc -arch=hexagon -filetype=obj %s | llvm-objdump -d - | FileCheck %s

        memw(gp+#hi_htc_version) = r3
#CHECK: 4880c300 { memw(gp+#0) = r3 }
        memw(gp+#HI) = r3
#CHECK: 4880c300 { memw(gp+#0) = r3 }
        r3 = memw(gp+#HI)
#CHECK: 4980c003 { r3 = memw(gp+#0) }
        memw(gp+#HI_x) = r3
#CHECK: 4880c300 { memw(gp+#0) = r3 }
        r3 = memw(gp+#HI_x)
#CHECK: 4980c003 { r3 = memw(gp+#0) }
        memw(gp+#hi) = r3
#CHECK: 4880c300 { memw(gp+#0) = r3 }
        r3 = memw(gp+#hi)
#CHECK: 4980c003 { r3 = memw(gp+#0) }
        memw(gp+#hi_x) = r3
#CHECK: 4880c300 { memw(gp+#0) = r3 }
        r3 = memw(gp+#hi_x)
#CHECK: 4980c003 { r3 = memw(gp+#0) }
        memw(gp+#lo) = r3
#CHECK: 4880c300 { memw(gp+#0) = r3 }
        r3 = memw(gp+#lo)
#CHECK: 4980c003 { r3 = memw(gp+#0) }
        memw(gp+#lo_x) = r3
#CHECK: 4880c300 { memw(gp+#0) = r3 }
        r3 = memw(gp+#lo_x)
#CHECK: 4980c003 { r3 = memw(gp+#0) }
        memw(gp+#LO) = r3
#CHECK: 4880c300 { memw(gp+#0) = r3 }
        r3 = memw(gp+#lo)
#CHECK: 4980c003 { r3 = memw(gp+#0) }
        memw(gp+#LO_x) = r3
#CHECK: 4880c300 { memw(gp+#0) = r3 }
        r3 = memw(gp+#LO_x)
#CHECK: 4980c003 { r3 = memw(gp+#0) }
        r16.h = #HI(0x405000)
#CHECK: 7230c040 { r16.h = #64 }
        r16.h = #HI (0x405000)
#CHECK: 7230c040 { r16.h = #64 }
        r16.h = #hi(0x405000)
#CHECK: 7230c040 { r16.h = #64 }
        r16.h = #hi (0x405000)
#CHECK: 7230c040 { r16.h = #64 }
        r16.l = #LO(0x405020)
#CHECK: 7170d020 { r16.l = #20512 }
        r16.l = #LO (0x405020)
#CHECK: 7170d020 { r16.l = #20512 }
        r16.l = #lo(0x405020)
#CHECK: 7170d020 { r16.l = #20512 }
        r16.l = #lo (0x405020)
#CHECK: 7170d020 { r16.l = #20512 }

{
  r19.h = #HI(-559030611)
  memw(r17+#0) = r19.new
}
# CHECK: 72f35ead { r19.h = #57005
# CHECK: a1b1d200   memw(r17+#0) = r19.new }

