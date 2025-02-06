; RUN: llc -mtriple=hexagon -verify-machineinstrs < %s | FileCheck %s

; Check for successful compilation.
; CHECK: sfcmp

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

@g0 = global <8 x i32> zeroinitializer, align 8

define void @fred(ptr %a0, ptr %a1) #0 {
b0:
  %v0 = load <8 x float>, ptr %a1, align 8
  %v1 = fcmp olt <8 x float> %v0, zeroinitializer
  %v2 = load <8 x float>, ptr %a0, align 8
  %v3 = fcmp olt <8 x float> %v2, zeroinitializer
  %v4 = and <8 x i1> %v1, %v3
  %v5 = zext <8 x i1> %v4 to <8 x i32>
  store <8 x i32> %v5, ptr @g0, align 8
  ret void
}

attributes #0 = { noinline norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length64b,+hvxv60" }
