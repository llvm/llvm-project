; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; The first argument must be a vector of pointers.
; CHECK: intrinsic argument 0 type (overload type 0) expected any pointer vector, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.experimental.vector.histogram.add.v4i32.i32(<4 x i32>, i32, <4 x i1>)
declare void @llvm.experimental.vector.histogram.add.v4i32.i32(<4 x i32>, i32, <4 x i1>)

; CHECK: intrinsic argument 0 type (overload type 0) expected any pointer vector, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.experimental.vector.histogram.uadd.sat.v4i32.i32(<4 x i32>, i32, <4 x i1>)
declare void @llvm.experimental.vector.histogram.uadd.sat.v4i32.i32(<4 x i32>, i32, <4 x i1>)

; CHECK: intrinsic argument 0 type (overload type 0) expected any pointer vector, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.experimental.vector.histogram.umin.v4i32.i32(<4 x i32>, i32, <4 x i1>)
declare void @llvm.experimental.vector.histogram.umin.v4i32.i32(<4 x i32>, i32, <4 x i1>)

; CHECK: intrinsic argument 0 type (overload type 0) expected any pointer vector, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.experimental.vector.histogram.umax.v4i32.i32(<4 x i32>, i32, <4 x i1>)
declare void @llvm.experimental.vector.histogram.umax.v4i32.i32(<4 x i32>, i32, <4 x i1>)

; The increment/update value must be a scalar integer.
; CHECK: intrinsic argument 1 type (overload type 1) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.experimental.vector.histogram.add.v4p0.v4i32(<4 x ptr>, <4 x i32>, <4 x i1>)
declare void @llvm.experimental.vector.histogram.add.v4p0.v4i32(<4 x ptr>, <4 x i32>, <4 x i1>)

; CHECK: intrinsic argument 1 type (overload type 1) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.experimental.vector.histogram.uadd.sat.v4p0.v4i32(<4 x ptr>, <4 x i32>, <4 x i1>)
declare void @llvm.experimental.vector.histogram.uadd.sat.v4p0.v4i32(<4 x ptr>, <4 x i32>, <4 x i1>)

; CHECK: intrinsic argument 1 type (overload type 1) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.experimental.vector.histogram.umin.v4p0.v4i32(<4 x ptr>, <4 x i32>, <4 x i1>)
declare void @llvm.experimental.vector.histogram.umin.v4p0.v4i32(<4 x ptr>, <4 x i32>, <4 x i1>)

; CHECK: intrinsic argument 1 type (overload type 1) expected any integer type, but got <4 x i32>
; CHECK-NEXT: declare void @llvm.experimental.vector.histogram.umax.v4p0.v4i32(<4 x ptr>, <4 x i32>, <4 x i1>)
declare void @llvm.experimental.vector.histogram.umax.v4p0.v4i32(<4 x ptr>, <4 x i32>, <4 x i1>)
