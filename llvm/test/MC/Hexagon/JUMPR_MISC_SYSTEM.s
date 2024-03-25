# RUN: llvm-mc --triple=hexagon --mv73 -filetype=obj %s | \
# RUN:    llvm-objdump --mcpu=hexagonv73 -d - | FileCheck %s
                         hintjr(R31)
# CHECK:      52bfc000 { hintjr(r31) }
                         trap0(#0xff)
# CHECK-NEXT: 5400df1c { trap0(#0xff) }
                         trap1(#0xff)
# CHECK-NEXT: 5480df1c { trap1(r0,#0xff) }
                         pause(#0xff)
# CHECK-NEXT: 5440df1c { pause(#0xff) }
                         R31=icdatar(R31)
# CHECK-NEXT: 55bfc01f { r31 = icdatar(r31) }
                         R31=ictagr(R31)
# CHECK-NEXT: 55ffc01f { r31 = ictagr(r31) }
                         ictagw(R31,R31)
# CHECK-NEXT: 55dfdf00 { ictagw(r31,r31) }
                         icinva(R31)
# CHECK-NEXT: 56dfc000 { icinva(r31) }
                         icinvidx(R31)
# CHECK-NEXT: 56dfc800 { icinvidx(r31) }
                         ickill
# CHECK-NEXT: 56c0d000 { ickill }
                         isync
# CHECK-NEXT: 57c0c002 { isync }
                         rte
# CHECK-NEXT: 57e0c000 { rte }
