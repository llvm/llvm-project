; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that this compiles successfully.
; CHECK: vdeal

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

@g0 = global <16 x i16> zeroinitializer, align 2

define void @fred(<16 x i32> %a0, <16 x i32> %a1) #0 {
b0:
  %v0 = icmp eq <16 x i32> %a0, %a1
  %v1 = select <16 x i1> %v0, <16 x i32> %a0, <16 x i32> zeroinitializer
  %v2 = trunc <16 x i32> %v1 to <16 x i16>
  store <16 x i16> %v2, <16 x i16>* @g0, align 2
  ret void
}

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv65" "target-features"="+hvx-length64b,+hvxv65" }
