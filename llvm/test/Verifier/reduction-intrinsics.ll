; RUN: not opt -S -passes=verify -disable-output < %s 2>&1 | FileCheck %s

; Reject a vector reduction with a non-vector argument.

; CHECK: intrinsic argument 0 type (overload type 0) expected any fp vector, but got float
; CHECK-NEXT: declare float @llvm.vector.reduce.fmax.f32(float)
declare float @llvm.vector.reduce.fmax.f32(float)

; CHECK: intrinsic argument 0 type (overload type 0) expected any integer vector, but got i32
; CHECK-NEXT: declare i32 @llvm.vector.reduce.smax.i32(i32)
declare i32 @llvm.vector.reduce.smax.i32(i32)

; Type mismatch for start value.
; CHECK: intrinsic argument 0 type (vector element of overload type 0) expected float (overload type 0 is <4 x float>), but got double
; CHECK-NEXT: declare float @llvm.vector.reduce.fadd.v4f32(double, <4 x float>)
declare float @llvm.vector.reduce.fadd.v4f32(double, <4 x float>)

; Wrong result type.
; CHECK: intrinsic return type (vector element of overload type 0) expected i32 (overload type 0 is <4 x i32>), but got i64
; CHECK-NEXT: declare i64 @llvm.vector.reduce.add.v4i32(<4 x i32>)
declare i64 @llvm.vector.reduce.add.v4i32(<4 x i32>)

; We should have the appropriate (either int or FP) type of argument
; for any vector reduction.
; CHECK: intrinsic argument 0 type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare float @llvm.vector.reduce.umin.v4f32(<4 x float>)
declare float @llvm.vector.reduce.umin.v4f32(<4 x float>)

; CHECK: intrinsic argument 0 type (overload type 0) expected any integer vector, but got <4 x ptr>
; CHECK-NEXT: declare ptr @llvm.vector.reduce.or.v4p0(<4 x ptr>)
declare ptr @llvm.vector.reduce.or.v4p0(<4 x ptr>)

; CHECK: intrinsic argument 1 type (overload type 0) expected any fp vector, but got <4 x i32>
; CHECK-NEXT: declare i32 @llvm.vector.reduce.fadd.v4i32(i32, <4 x i32>)
declare i32 @llvm.vector.reduce.fadd.v4i32(i32, <4 x i32>)

; CHECK: intrinsic argument 0 type (overload type 0) expected any fp vector, but got <4 x ptr>
; CHECK-NEXT: declare ptr @llvm.vector.reduce.fmin.v4p0(<4 x ptr>)
declare ptr @llvm.vector.reduce.fmin.v4p0(<4 x ptr>)
