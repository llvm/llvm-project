; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Check that this doesn't crash.
; CHECK: vmem

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define dllexport void @f0(ptr %a0, i32 %a1, i32 %a2, <32 x i32> %a3) local_unnamed_addr #0 {
b0:
  %v0 = add nuw nsw i32 0, 96
  %v1 = getelementptr inbounds i32, ptr %a0, i32 %v0
  %v3 = add nuw nsw i32 0, 225
  %v4 = getelementptr inbounds i32, ptr %a0, i32 %v3
  %v5 = add nuw nsw i32 0, 226
  %v6 = getelementptr inbounds i32, ptr %a0, i32 %v5
  br label %b1

b1:                                               ; preds = %b1, %b0
  br i1 undef, label %b2, label %b1

b2:                                               ; preds = %b1
  store <32 x i32> %a3, ptr %v1, align 4
  store i32 %a1, ptr %v4, align 4
  store i32 %a2, ptr %v6, align 4
  ret void
}

attributes #0 = { "target-features"="+hvxv66,+hvx-length128b" }
