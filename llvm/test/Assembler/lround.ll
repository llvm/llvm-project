; Validate that vector types are accepted for llvm.lround/llvm.llround intrinsic
; RUN: llvm-as < %s | llvm-dis |  FileCheck %s

define <2 x i32> @intrinsic_lround_v2i32_v2f32(<2 x float> %arg) {
  ;CHECK: %res = tail call <2 x i32> @llvm.lround.v2i32.v2f32(<2 x float> %arg)
  %res = tail call <2 x i32> @llvm.lround.v2i32.v2f32(<2 x float> %arg)
  ret <2 x i32> %res
}

define <2 x i32> @intrinsic_llround_v2i32_v2f32(<2 x float> %arg) {
  ;CHECK: %res = tail call <2 x i32> @llvm.llround.v2i32.v2f32(<2 x float> %arg)
  %res = tail call <2 x i32> @llvm.llround.v2i32.v2f32(<2 x float> %arg)
  ret <2 x i32> %res
}

define <2 x i64> @intrinsic_lround_v2i64_v2f32(<2 x float> %arg) {
  ;CHECK: %res = tail call <2 x i64> @llvm.lround.v2i64.v2f32(<2 x float> %arg)
  %res = tail call <2 x i64> @llvm.lround.v2i64.v2f32(<2 x float> %arg)
  ret <2 x i64> %res
}

define <2 x i64> @intrinsic_llround_v2i64_v2f32(<2 x float> %arg) {
  ;CHECK: %res = tail call <2 x i64> @llvm.llround.v2i64.v2f32(<2 x float> %arg)
  %res = tail call <2 x i64> @llvm.llround.v2i64.v2f32(<2 x float> %arg)
  ret <2 x i64> %res
}
