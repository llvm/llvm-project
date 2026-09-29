; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Check that this testcase doesn't crash.
; CHECK: sfcmp

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define void @fred(ptr %a0, ptr %a1, ptr %a2) #0 {
b0:
  %v0 = load <16 x float>, ptr %a0, align 128
  %v1 = fcmp olt <16 x float> zeroinitializer, %v0
  %v2 = load <16 x i16>, ptr %a1, align 128
  %v3 = select <16 x i1> %v1, <16 x i16> %v2, <16 x i16> zeroinitializer
  %v4 = sext <16 x i16> %v3 to <16 x i32>
  store <16 x i32> %v4, ptr %a2, align 128
  ret void
}

attributes #0 = { noinline norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length64b" }
