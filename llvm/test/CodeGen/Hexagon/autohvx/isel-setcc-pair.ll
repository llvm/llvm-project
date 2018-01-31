; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that a setcc of a vector pair is handled (without crashing).
; CHECK: vcmp

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

; Function Attrs: nounwind
define hidden fastcc void @fred(i32 %a0) #0 {
b1:
  %v2 = insertelement <32 x i32> undef, i32 %a0, i32 0
  %v3 = shufflevector <32 x i32> %v2, <32 x i32> undef, <32 x i32> zeroinitializer
  %v4 = icmp eq <32 x i32> %v3, undef
  %v5 = and <32 x i1> undef, %v4
  br label %b6

b6:                                               ; preds = %b1
  %v7 = extractelement <32 x i1> %v5, i32 22
  br i1 %v7, label %b8, label %b9

b8:                                               ; preds = %b6
  unreachable

b9:                                               ; preds = %b6
  unreachable
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }
