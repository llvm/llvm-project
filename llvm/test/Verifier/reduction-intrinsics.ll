; RUN: not opt -S -passes=verify < %s 2>&1 | FileCheck %s

; Reject a vector reduction with a non-vector argument.

define float @reduce_vector_not_vec_arg(float %x) {
; CHECK: Intrinsic has incorrect argument type!
  %r0 = call float @llvm.vector.reduce.fmax.f32(float %x)
  ret float %r0
}

define i32 @reduce_vector_not_vec_arg2(i32 %x) {
; CHECK: Intrinsic has incorrect argument type!
  %r0 = call i32 @llvm.vector.reduce.smax.i32(i32 %x)
  ret i32 %r0
}

; Type mismatch for start value.

define float @fadd_match_arg_types(<4 x float> %x) {
; CHECK: Intrinsic has incorrect argument type!
  %r = call float @llvm.vector.reduce.fadd.v4f32(double 0.0, <4 x float> %x)
  ret float %r
}

; Wrong result type.

define i64 @result_too_wide(<4 x i32> %x) {
; CHECK: Intrinsic has incorrect return type!
  %r = call i64 @llvm.vector.reduce.add.v4i32(<4 x i32> %x)
  ret i64 %r
}

; We should have the appropriate (either int or FP) type of argument
; for any vector reduction.

define float @not_float_reduce(<4 x float> %x) {
; CHECK: Intrinsic has incorrect argument type!
  %r = call float @llvm.vector.reduce.umin.v4f32(<4 x float> %x)
  ret float %r
}

define ptr @not_pointer_reduce(<4 x ptr> %x) {
; CHECK: Intrinsic has incorrect argument type!
  %r = call ptr @llvm.vector.reduce.or.v4p0(<4 x ptr> %x)
  ret ptr %r
}

define i32 @not_integer_reduce(<4 x i32> %x) {
; CHECK: Intrinsic has incorrect argument type!
  %r = call i32 @llvm.vector.reduce.fadd.v4i32(i32 0, <4 x i32> %x)
  ret i32 %r
}

define ptr @not_pointer_reduce2(<4 x ptr> %x) {
; CHECK: Intrinsic has incorrect argument type!
  %r = call ptr @llvm.vector.reduce.fmin.v4p0(<4 x ptr> %x)
  ret ptr %r
}

declare float @llvm.vector.reduce.umin.v4f32(<4 x float>)
declare ptr @llvm.vector.reduce.or.v4p0(<4 x ptr>)
declare i32 @llvm.vector.reduce.fadd.v4i32(i32, <4 x i32>)
declare float @llvm.vector.reduce.fadd.v4f32(double, <4 x float>)
declare ptr @llvm.vector.reduce.fmin.v4p0(<4 x ptr>)
declare float @llvm.vector.reduce.fmax.f32(float)
declare i32 @llvm.vector.reduce.smax.i32(i32)
declare i64 @llvm.vector.reduce.add.v4i32(<4 x i32>)
