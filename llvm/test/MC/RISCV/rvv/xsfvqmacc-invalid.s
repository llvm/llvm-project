# RUN: not llvm-mc -triple=riscv64 -show-encoding -mattr=+v,+xsfvqmaccqoq,+xsfvqmaccdod %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

sf.vqmaccu.2x8x2 v8, v8, v20
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group{{$}}
# CHECK-ERROR-LABEL: sf.vqmaccu.2x8x2 v8, v8, v20

sf.vqmacc.2x8x2 v8, v8, v20
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group{{$}}
# CHECK-ERROR-LABEL: sf.vqmacc.2x8x2 v8, v8, v20

sf.vqmaccus.2x8x2 v8, v8, v20
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group{{$}}
# CHECK-ERROR-LABEL: sf.vqmaccus.2x8x2 v8, v8, v20

sf.vqmaccsu.2x8x2 v8, v8, v20
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group{{$}}
# CHECK-ERROR-LABEL: sf.vqmaccsu.2x8x2 v8, v8, v20

sf.vqmaccu.4x8x4 v8, v8, v20
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group{{$}}
# CHECK-ERROR-LABEL: sf.vqmaccu.4x8x4 v8, v8, v20

sf.vqmacc.4x8x4 v8, v8, v20
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group{{$}}
# CHECK-ERROR-LABEL: sf.vqmacc.4x8x4 v8, v8, v20

sf.vqmaccus.4x8x4 v8, v8, v20
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group{{$}}
# CHECK-ERROR-LABEL: sf.vqmaccus.4x8x4 v8, v8, v20

sf.vqmaccsu.4x8x4 v8, v8, v20
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group{{$}}
# CHECK-ERROR-LABEL: sf.vqmaccsu.4x8x4 v8, v8, v20

sf.vqmaccu.4x8x4 v8, v4, v8
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group{{$}}
# CHECK-ERROR-LABEL: sf.vqmaccu.4x8x4 v8, v4, v8

sf.vqmacc.4x8x4 v8, v4, v8
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group{{$}}
# CHECK-ERROR-LABEL: sf.vqmacc.4x8x4 v8, v4, v8

sf.vqmaccus.4x8x4 v8, v4, v8
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group{{$}}
# CHECK-ERROR-LABEL: sf.vqmaccus.4x8x4 v8, v4, v8

sf.vqmaccsu.4x8x4 v8, v4, v8
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group{{$}}
# CHECK-ERROR-LABEL: sf.vqmaccsu.4x8x4 v8, v4, v8
