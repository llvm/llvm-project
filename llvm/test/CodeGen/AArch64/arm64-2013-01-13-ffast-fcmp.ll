; RUN: llc < %s -mtriple=arm64-apple-ios7.0.0 -aarch64-neon-syntax=apple | FileCheck %s
; RUN: llc < %s -mtriple=arm64-apple-ios7.0.0 -aarch64-neon-syntax=apple -fp-contract=fast | FileCheck %s --check-prefix=FAST

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32:64-S128"

;FAST-LABEL: _Z9example25v:
;FAST: fcmgt.4s
;FAST: ret

;CHECK-LABEL: _Z9example25v:
;CHECK: fcmgt.4s
;CHECK: ret

define <4 x i32> @_Z9example25v( <4 x float> %N0,  <4 x float> %N1) {
  %A = fcmp olt <4 x float> %N0, %N1
  %B = zext <4 x i1> %A to <4 x i32>
  ret <4 x i32> %B
}
