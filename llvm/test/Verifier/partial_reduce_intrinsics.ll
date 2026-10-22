; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

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
