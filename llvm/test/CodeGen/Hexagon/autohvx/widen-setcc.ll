; RUN: llc -mtriple=hexagon -hexagon-hvx-widen=32 < %s | FileCheck %s

; Make sure that this doesn't crash.
; CHECK-LABEL: f0:
; CHECK: vmem

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define dllexport void @f0(ptr %a0, <16 x i16> %a1) local_unnamed_addr #0 {
b0:
  %v0 = getelementptr i16, ptr %a0, i32 undef
  %v2 = load <16 x i16>, ptr undef, align 2
  %v3 = icmp sgt <16 x i16> zeroinitializer, %v2
  %v4 = select <16 x i1> %v3, <16 x i16> %a1, <16 x i16> %v2
  store <16 x i16> %v4, ptr %v0, align 2
  ret void
}

attributes #0 = { "target-features"="+hvxv66,+hvx-length128b" }
