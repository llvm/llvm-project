; RUN: llc -march=hexagon < %s | FileCheck %s

; This used to crash in SimplifyDemandedBits due to a type mismatch
; caused by a missing bitcast in vectorizing mul.
; CHECK: vmpy

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define void @fred(i16 signext %a0, <32 x i16>* %a1, <32 x i16> %a3) #0 {
b1:
  %v4 = add i16 undef, %a0
  br i1 undef, label %b11, label %b5

b5:                                               ; preds = %b1
  %v6 = insertelement <32 x i16> undef, i16 %v4, i32 0
  %v7 = shufflevector <32 x i16> %v6, <32 x i16> undef, <32 x i32> zeroinitializer
  %v8 = add <32 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15, i16 16, i16 17, i16 18, i16 19, i16 20, i16 21, i16 22, i16 23, i16 24, i16 25, i16 26, i16 27, i16 28, i16 29, i16 30, i16 31>, <i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256>
  %v9 = mul <32 x i16> %v8, %a3
  %v10 = add <32 x i16> %v7, %v9
  store <32 x i16> %v10, <32 x i16>* %a1, align 2
  ret void

b11:                                              ; preds = %b1
  unreachable
}

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length64b,+hvxv60" }
