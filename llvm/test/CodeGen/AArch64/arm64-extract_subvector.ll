; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

; Extract of an upper half of a vector is an "ext.16b v0, v0, v0, #8" insn.

define <8 x i8> @v8i8(<16 x i8> %a) nounwind {
; CHECK: v8i8
; CHECK: ext.16b v0, v0, v0, #8
; CHECK: ret
  %ret = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32>  <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <8 x i8> %ret
}

define <4 x i16> @v4i16(<8 x i16> %a) nounwind {
; CHECK-LABEL: v4i16:
; CHECK: ext.16b v0, v0, v0, #8
; CHECK: ret
  %ret = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32>  <i32 4, i32 5, i32 6, i32 7>
  ret <4 x i16> %ret
}

define <2 x i32> @v2i32(<4 x i32> %a) nounwind {
; CHECK-LABEL: v2i32:
; CHECK: ext.16b v0, v0, v0, #8
; CHECK: ret
  %ret = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32>  <i32 2, i32 3>
  ret <2 x i32> %ret
}

define <1 x i64> @v1i64(<2 x i64> %a) nounwind {
; CHECK-LABEL: v1i64:
; CHECK: ext.16b v0, v0, v0, #8
; CHECK: ret
  %ret = shufflevector <2 x i64> %a, <2 x i64> %a, <1 x i32>  <i32 1>
  ret <1 x i64> %ret
}

define <2 x float> @v2f32(<4 x float> %a) nounwind {
; CHECK-LABEL: v2f32:
; CHECK: ext.16b v0, v0, v0, #8
; CHECK: ret
  %ret = shufflevector <4 x float> %a, <4 x float> %a, <2 x i32>  <i32 2, i32 3>
  ret <2 x float> %ret
}

define <1 x double> @v1f64(<2 x double> %a) nounwind {
; CHECK-LABEL: v1f64:
; CHECK: ext.16b v0, v0, v0, #8
; CHECK: ret
  %ret = shufflevector <2 x double> %a, <2 x double> %a, <1 x i32>  <i32 1>
  ret <1 x double> %ret
}
