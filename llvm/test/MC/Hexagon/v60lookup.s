#RUN: llvm-mc -triple=hexagon -mcpu=hexagonv60 -filetype=obj %s | \
#RUN: llvm-objdump -triple=hexagon -mcpu=hexagonv60 -d - | \
#RUN: FileCheck %s

                    V31.b = vlut32(V29.b, V15.b, R1)
# CHECK: 1b79fd3f { v31.b = vlut32(v29.b,v15.b,r1) }
                    V31.b |= vlut32(V29.b, V15.b, R2)
# CHECK: 1b7afdbf { v31.b |= vlut32(v29.b,v15.b,r2) }
                    V31:30.h = vlut16(V29.b, V15.h, R3)
# CHECK: 1b7bfdde { v31:30.h = vlut16(v29.b,v15.h,r3) }
                    v31:30.h |= vlut16(v2.b, v9.h, r4)
# CHECK: 1b4ce2fe { v31:30.h |= vlut16(v2.b,v9.h,r4) }
                    v31.w = vinsert(r4)
# CHECK: 19a4e03f { v31.w = vinsert(r4) }
