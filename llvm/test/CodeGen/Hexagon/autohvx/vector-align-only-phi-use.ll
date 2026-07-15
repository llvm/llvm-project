; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; One of the loads is only used in a PHI instruction. Make sure the PHI use
; still counts as a user of the load (and that the load is not removed).

; CHECK-LABEL: f0:
; CHECK: = vmem({{.*}})
; CHECK: = vmem({{.*}})

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

declare <16 x i32> @llvm.hexagon.V6.vmux(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vmpyiewuh.acc(<16 x i32>, <16 x i32>, <16 x i32>) #0
declare <16 x i32> @llvm.hexagon.V6.vand(<16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.vgtuw.and(<64 x i1>, <16 x i32>, <16 x i32>) #0
declare <64 x i1> @llvm.hexagon.V6.pred.or(<64 x i1>, <64 x i1>) #0

define <16 x i32> @f0(ptr %a0, i32 %a1) local_unnamed_addr #1 {
b0:
  %v0 = getelementptr inbounds i8, ptr %a0, i32 576
  br label %b1

b1:                                               ; preds = %b4, %b0
  %v3 = phi i32 [ 0, %b0 ], [ %v23, %b4 ]
  %v4 = phi <16 x i32> [ poison, %b0 ], [ %v22, %b4 ]
  br i1 poison, label %b2, label %b3

b2:                                               ; preds = %b1
  %v5 = getelementptr inbounds <16 x i32>, ptr %a0, i32 %v3
  %v6 = load <16 x i32>, ptr %v5, align 64
  %v7 = getelementptr inbounds <16 x i32>, ptr %v0, i32 %v3
  %v8 = load <16 x i32>, ptr %v7, align 64
  %v9 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiewuh.acc(<16 x i32> poison, <16 x i32> %v6, <16 x i32> %v6)
  br label %b4

b3:                                               ; preds = %b1
  br label %b4

b4:                                               ; preds = %b3, %b2
  %v10 = phi <16 x i32> [ %v9, %b2 ], [ poison, %b3 ]
  %v11 = phi <16 x i32> [ %v8, %b2 ], [ poison, %b3 ]
  %v12 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<64 x i1> poison, <16 x i32> %v10, <16 x i32> %v4)
  %v13 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<64 x i1> poison, <16 x i32> %v11, <16 x i32> poison)
  %v14 = or i32 %v3, 1
  %v15 = getelementptr inbounds <16 x i32>, ptr %v0, i32 %v14
  %v16 = load <16 x i32>, ptr %v15, align 64
  %v17 = tail call <16 x i32> @llvm.hexagon.V6.vmpyiewuh.acc(<16 x i32> poison, <16 x i32> %v13, <16 x i32> poison)
  %v18 = tail call <16 x i32> @llvm.hexagon.V6.vand(<16 x i32> %v12, <16 x i32> poison)
  %v19 = tail call <64 x i1> @llvm.hexagon.V6.vgtuw.and(<64 x i1> poison, <16 x i32> %v17, <16 x i32> poison)
  %v20 = tail call <64 x i1> @llvm.hexagon.V6.pred.or(<64 x i1> %v19, <64 x i1> poison)
  %v21 = tail call <64 x i1> @llvm.hexagon.V6.pred.or(<64 x i1> %v20, <64 x i1> poison)
  %v22 = tail call <16 x i32> @llvm.hexagon.V6.vmux(<64 x i1> %v21, <16 x i32> poison, <16 x i32> %v12)
  %v23 = add nuw nsw i32 %v3, 2
  %v24 = icmp slt i32 %v23, %a1
  br i1 %v24, label %b1, label %b5, !llvm.loop !1

b5:                                               ; preds = %b4
  ret <16 x i32> %v22
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #1 = { "target-features"="+hvx-length64b,+hvxv65,+v65,-long-calls" }

!llvm.linker.options = !{!0}

!0 = !{!"$.str.3", !".rodata.str1.1"}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.mustprogress"}
