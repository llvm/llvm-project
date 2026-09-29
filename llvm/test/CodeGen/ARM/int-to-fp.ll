; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10.0.0"

; CHECK: sint_to_fp
; CHECK: vmovl.s16
; CHECK: vcvt.f32.s32
define <4 x float> @sint_to_fp(<4 x i16> %x) nounwind ssp {
  %a = sitofp <4 x i16> %x to <4 x float>
  ret <4 x float> %a
}

; CHECK: uint_to_fp
; CHECK: vmovl.u16
; CHECK: vcvt.f32.u32
define <4 x float> @uint_to_fp(<4 x i16> %x) nounwind ssp {
  %a = uitofp <4 x i16> %x to <4 x float>
  ret <4 x float> %a
}
