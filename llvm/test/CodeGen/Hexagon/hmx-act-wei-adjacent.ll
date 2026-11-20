; RUN: llc -march=hexagon < %s | FileCheck %s

; These two should be adjacent.
; CHECK: activation.hf = mxmem
; CHECK-NEXT: weight.hf = mxmem

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define dllexport void @f0(i1 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4) local_unnamed_addr #0 {
b0:
  %v0 = add nsw i32 0, 1
  %v1 = sub nuw nsw i32 4, 0
  %v2 = select i1 %a0, i32 4, i32 %v1
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v3 = phi i32 [ 0, %b0 ], [ %v13, %b1 ]
  %v4 = mul i32 %v3, -8
  %v5 = add i32 %v4, %a1
  %v6 = icmp slt i32 %v5, 8
  %v7 = select i1 %v6, i32 %v5, i32 8
  %v8 = shl nsw i32 %v7, 2
  %v9 = mul i32 %v8, %v2
  %v10 = shl i32 %v9, 6
  %v11 = mul i32 %v10, %v0
  %v12 = add nsw i32 %v11, -1
  tail call void @llvm.hexagon.M8.mxmem.blk.sm.act.hf(i32 %a2, i32 %a3)
  tail call void @llvm.hexagon.M8.mxmem.wei.hf(i32 %a4, i32 %v12)
  %v13 = add nuw nsw i32 %v3, 1
  br label %b1
}

; Function Attrs: nounwind writeonly
declare void @llvm.hexagon.M8.mxmem.wei.hf(i32, i32) #1

; Function Attrs: nounwind writeonly
declare void @llvm.hexagon.M8.mxmem.blk.sm.act.hf(i32, i32) #1

attributes #0 = { "target-features"="+hmx,-packets" }
attributes #1 = { nounwind writeonly }
