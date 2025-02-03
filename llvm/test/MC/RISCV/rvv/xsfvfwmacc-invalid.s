# RUN: not llvm-mc -triple=riscv64 -show-encoding -mattr=+v,+xsfvfwmaccqqq %s 2>&1 \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR

sf.vfwmacc.4x4x4 v8, v8, v20
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group{{$}}
# CHECK-ERROR-LABEL: sf.vfwmacc.4x4x4 v8, v8, v20{{$}}

sf.vfwmacc.4x4x4 v8, v4, v8
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group{{$}}
# CHECK-ERROR-LABEL: sf.vfwmacc.4x4x4 v8, v4, v8{{$}}
