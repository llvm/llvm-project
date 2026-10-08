; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; The operands of the integer partial reduction must be vectors of integers.
; CHECK: intrinsic argument 0 type (overload type 0) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare <4 x float> @llvm.vector.partial.reduce.add.v4f32.v4f32(<4 x float>, <4 x float>)
declare <4 x float> @llvm.vector.partial.reduce.add.v4f32.v4f32(<4 x float>, <4 x float>)

; CHECK: intrinsic argument 1 type (overload type 1) expected any integer vector, but got <4 x float>
; CHECK-NEXT: declare <4 x float> @llvm.vector.partial.reduce.add.v4i32.v4f32(<4 x i32>, <4 x float>)
declare <4 x float> @llvm.vector.partial.reduce.add.v4i32.v4f32(<4 x i32>, <4 x float>)

; The operands of the floating-point partial reduction must be vectors of floats.
; CHECK: intrinsic argument 0 type (overload type 0) expected any fp vector, but got <4 x i32>
; CHECK-NEXT: declare <4 x i32> @llvm.vector.partial.reduce.fadd.v4i32.v4i32(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.vector.partial.reduce.fadd.v4i32.v4i32(<4 x i32>, <4 x i32>)

; CHECK: intrinsic argument 0 type (overload type 0) expected any fp vector, but got float
; CHECK-NEXT: declare float @llvm.vector.partial.reduce.fadd.f32.f32(float, float)
declare float @llvm.vector.partial.reduce.fadd.f32.f32(float, float)

; CHECK: intrinsic argument 1 type (overload type 1) expected any fp vector, but got float
; CHECK-NEXT: declare <4 x float> @llvm.vector.partial.reduce.fadd.v4f32.f32(<4 x float>, float)
declare <4 x float> @llvm.vector.partial.reduce.fadd.v4f32.f32(<4 x float>, float)

; The element type of the input vector must match the accumulator.

declare <4 x i32> @llvm.vector.partial.reduce.add.v4i32.v8i64(<4 x i32>, <8 x i64>)

define <4 x i32> @partial_reduce_add_mismatched_elt(<4 x i32> %a, <8 x i64> %b) {
; CHECK: The element type of the input vector must match the element type of the accumulator vector.
; CHECK-NEXT:   %r = call <4 x i32> @llvm.vector.partial.reduce.add.v4i32.v8i64(<4 x i32> %a, <8 x i64> %b)
  %r = call <4 x i32> @llvm.vector.partial.reduce.add.v4i32.v8i64(<4 x i32> %a, <8 x i64> %b)
  ret <4 x i32> %r
}

declare <4 x float> @llvm.vector.partial.reduce.fadd.v4f32.v8f64(<4 x float>, <8 x double>)

define <4 x float> @partial_reduce_fadd_mismatched_elt(<4 x float> %a, <8 x double> %b) {
; CHECK: The element type of the input vector must match the element type of the accumulator vector.
; CHECK-NEXT:   %r = call <4 x float> @llvm.vector.partial.reduce.fadd.v4f32.v8f64(<4 x float> %a, <8 x double> %b)
  %r = call <4 x float> @llvm.vector.partial.reduce.fadd.v4f32.v8f64(<4 x float> %a, <8 x double> %b)
  ret <4 x float> %r
}
