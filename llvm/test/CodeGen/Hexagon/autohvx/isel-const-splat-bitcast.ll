; RUN: llc -march=hexagon -verify-machineinstrs < %s | FileCheck %s

; The generation of a constant vector in the selection step resulted in
; a VSPLAT, which, deeper in the expression tree had an unrelated BITCAST.
; That bitcast was erroneously removed by the constant vector selection
; function, and caused a selection error due to a type mismatch.
;
; Make sure this compiles successfully.
; CHECK: vsplat

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

@g0 = global <8 x i16> zeroinitializer, align 2

define i32 @fred() #0 {
b0:
  %v1 = load <8 x i16>, ptr @g0, align 2
  %v2 = icmp sgt <8 x i16> %v1, <i16 11, i16 11, i16 11, i16 11, i16 11, i16 11, i16 11, i16 11>
  %v3 = zext <8 x i1> %v2 to <8 x i32>
  %v4 = add nuw nsw <8 x i32> zeroinitializer, %v3
  %v5 = add nuw nsw <8 x i32> %v4, zeroinitializer
  %v6 = shufflevector <8 x i32> %v5, <8 x i32> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %v7 = add nuw nsw <8 x i32> %v5, %v6
  %v8 = extractelement <8 x i32> %v7, i32 0
  %v9 = add nuw nsw i32 %v8, 0
  %v10 = add nuw nsw i32 %v9, 0
  %v11 = add nuw nsw i32 %v10, 0
  %v12 = icmp ult i32 %v11, 5
  br i1 %v12, label %b13, label %b14

b13:                                              ; preds = %b0
  ret i32 %v11

b14:                                              ; preds = %b0
  ret i32 0
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }
