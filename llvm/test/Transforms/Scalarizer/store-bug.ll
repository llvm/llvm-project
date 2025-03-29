; RUN: opt -passes='function(scalarizer<load-store>)' -S < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; This input caused the scalarizer not to clear cached results
; properly.
;
; Any regressions should trigger an assert in the scalarizer.

define void @func(<4 x float> %val, ptr %ptr) {
  store <4 x float> %val, ptr %ptr
  ret void
; CHECK: store float %val.i0, ptr %ptr, align 16
; CHECK: store float %val.i1, ptr %ptr.i1, align 4
; CHECK: store float %val.i2, ptr %ptr.i2, align 8
; CHECK: store float %val.i3, ptr %ptr.i3, align 4
}

define void @func.copy(<4 x float> %val, ptr %ptr) {
  store <4 x float> %val, ptr %ptr
  ret void
; CHECK: store float %val.i0, ptr %ptr, align 16
; CHECK: store float %val.i1, ptr %ptr.i1, align 4
; CHECK: store float %val.i2, ptr %ptr.i2, align 8
; CHECK: store float %val.i3, ptr %ptr.i3, align 4
}
