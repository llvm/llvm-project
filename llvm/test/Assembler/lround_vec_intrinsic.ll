; Validate that vector types are accepted for llvm.lround/llvm.llround intrinsic
; RUN: llvm-as < %s -disable-output 2>&1| FileCheck -allow-empty %s

; CHECK-NOT: assembly parsed, but does not verify as correct
; CHECK-NOT: Intrinsic does not support vectors

define <2 x i32> @intrinsic_lround_v2i32_v2f32(<2 x float> %arg) {
entry:
  %0 = tail call <2 x i32> @llvm.lround.v2i32.v2f32(<2 x float> %arg)
  ret <2 x i32> %0
}

define <2 x i32> @intrinsic_llround_v2i32_v2f32(<2 x float> %arg) {
entry:
  %0 = tail call <2 x i32> @llvm.llround.v2i32.v2f32(<2 x float> %arg)
  ret <2 x i32> %0
}

define <2 x i64> @intrinsic_lround_v2i64_v2f32(<2 x float> %arg) {
entry:
  %0 = tail call <2 x i64> @llvm.lround.v2i64.v2f32(<2 x float> %arg)
  ret <2 x i64> %0
}

define <2 x i64> @intrinsic_llround_v2i64_v2f32(<2 x float> %arg) {
entry:
  %0 = tail call <2 x i64> @llvm.llround.v2i64.v2f32(<2 x float> %arg)
  ret <2 x i64> %0
}
