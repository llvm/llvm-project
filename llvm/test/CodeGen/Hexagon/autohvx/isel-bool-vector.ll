; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that this testcase doesn't crash.
; CHECK: sfcmp

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define void @fred() #0 {
b0:
  %v1 = fcmp olt <16 x float> zeroinitializer, undef
  %v2 = select <16 x i1> %v1, <16 x i16> undef, <16 x i16> zeroinitializer
  %v3 = sext <16 x i16> %v2 to <16 x i32>
  store <16 x i32> %v3, <16 x i32>* undef, align 128
  unreachable
}

attributes #0 = { noinline norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length64b" }
