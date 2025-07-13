# RUN: not llvm-mc -triple=riscv32 --mattr=+v,+xsfvcp %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: not llvm-mc -triple=riscv64 --mattr=+v,+xsfvcp %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

sf.vc.v.vvw 0x3, v0, v2, v0
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group{{$}}
# CHECK-ERROR-LABEL: sf.vc.v.vvw 0x3, v0, v2, v0{{$}}

sf.vc.v.xvw 0x3, v0, v0, a1
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group{{$}}
# CHECK-ERROR-LABEL: sf.vc.v.xvw 0x3, v0, v0, a1{{$}}

sf.vc.v.ivw 0x3, v0, v0, 15
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group{{$}}
# CHECK-ERROR-LABEL: sf.vc.v.ivw 0x3, v0, v0, 15{{$}}

sf.vc.v.fvw 0x1, v0, v0, fa1
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group{{$}}
# CHECK-ERROR-LABEL: sf.vc.v.fvw 0x1, v0, v0, fa1{{$}}
