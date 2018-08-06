; RUN: llc -march=hexagon < %s | FileCheck %s

; This code causes any_extend_vector_inreg to appear in the selection DAG.
; Make sure that it is handled instead of crashing.
; CHECK: vmem

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define hidden fastcc void @fred() #0 {
b0:
  %v1 = load i16, i16* undef, align 2
  %v2 = insertelement <16 x i16> undef, i16 %v1, i32 15
  %v3 = zext <16 x i16> %v2 to <16 x i32>
  %v4 = shl nuw <16 x i32> %v3, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  store <16 x i32> %v4, <16 x i32>* undef, align 4
  unreachable
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }
