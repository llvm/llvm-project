; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; This used to crash because of calling isSafeToMoveBeforeInBB with source
; and target in different blocks.
; Check that this compiles successfully, and that two loads are created
; (for users in a different block).

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1124:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define dso_local <32 x i32> @f0(i32 %a0, i32 %a1) local_unnamed_addr #0 {
; CHECK-LABEL: f0:
; CHECK:     = vmem({{.*}})
; CHECK:     = vmem({{.*}})
b0:
  br label %b1

b1:                                               ; preds = %b0
  %v0 = mul nsw i32 -4, %a0
  %v1 = getelementptr inbounds i8, ptr null, i32 %v0
  %v2 = getelementptr inbounds i8, ptr %v1, i32 -64
  %v4 = load <16 x i32>, ptr %v2, align 64
  %v5 = getelementptr inbounds i8, ptr %v1, i32 64
  %v7 = load <16 x i32>, ptr %v5, align 64
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v8 = phi <32 x i32> [ poison, %b1 ], [ %v17, %b2 ]
  %v9 = phi i32 [ %a1, %b1 ], [ %v18, %b2 ]
  %v10 = tail call <16 x i32> @llvm.hexagon.V6.vlalignb(<16 x i32> poison, <16 x i32> %v4, i32 poison)
  %v11 = tail call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %v7, <16 x i32> poison, i32 poison)
  %v12 = tail call <32 x i32> @llvm.hexagon.V6.vmpyubv(<16 x i32> %v10, <16 x i32> poison)
  %v13 = tail call <32 x i32> @llvm.hexagon.V6.vmpyubv(<16 x i32> %v11, <16 x i32> poison)
  %v14 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v12)
  %v15 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %v13)
  %v16 = tail call <32 x i32> @llvm.hexagon.V6.vadduhw(<16 x i32> %v14, <16 x i32> %v15)
  %v17 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.dv(<32 x i32> %v8, <32 x i32> %v16)
  %v18 = add nsw i32 %v9, -1
  %v19 = icmp ugt i32 %v18, 1
  br i1 %v19, label %b2, label %b3

b3:                                               ; preds = %b2
  %v20 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.dv(<32 x i32> %v17, <32 x i32> %v16)
  ret <32 x i32> %v20
}

declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #1
declare <16 x i32> @llvm.hexagon.V6.vlalignb(<16 x i32>, <16 x i32>, i32) #1
declare <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32>, <16 x i32>, i32) #1
declare <32 x i32> @llvm.hexagon.V6.vmpyubv(<16 x i32>, <16 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vaddw.dv(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vadduhw(<16 x i32>, <16 x i32>) #1

attributes #0 = { "target-features"="+hvx-length64b,+hvxv66,+v66,-long-calls" }
attributes #1 = { nounwind memory(none) }
