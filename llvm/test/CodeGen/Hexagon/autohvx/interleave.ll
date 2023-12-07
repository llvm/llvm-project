; RUN: opt -march=hexagon -hexagon-autohvx -passes=loop-vectorize -S < %s | FileCheck %s
; Check that the loop has been interleaved.
; CHECK: store <64 x i32> %interleaved.vec

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define void @f0(ptr noalias nocapture %a0, ptr noalias nocapture readonly %a1, i32 %a2) #0 {
b0:
  %v0 = icmp eq i32 %a2, 0
  br i1 %v0, label %b3, label %b1

b1:                                               ; preds = %b0
  br label %b4

b2:                                               ; preds = %b4
  br label %b3

b3:                                               ; preds = %b2, %b0
  ret void

b4:                                               ; preds = %b4, %b1
  %v1 = phi i32 [ %v13, %b4 ], [ 0, %b1 ]
  %v2 = getelementptr inbounds i32, ptr %a1, i32 %v1
  %v3 = load i32, ptr %v2, align 4, !tbaa !1
  %v4 = getelementptr inbounds i32, ptr %a0, i32 %v1
  %v5 = load i32, ptr %v4, align 4, !tbaa !1
  %v6 = add nsw i32 %v5, %v3
  store i32 %v6, ptr %v4, align 4, !tbaa !1
  %v7 = or i32 %v1, 1
  %v8 = getelementptr inbounds i32, ptr %a1, i32 %v7
  %v9 = load i32, ptr %v8, align 4, !tbaa !1
  %v10 = getelementptr inbounds i32, ptr %a0, i32 %v7
  %v11 = load i32, ptr %v10, align 4, !tbaa !1
  %v12 = add nsw i32 %v11, %v9
  store i32 %v12, ptr %v10, align 4, !tbaa !1
  %v13 = add nuw nsw i32 %v1, 2
  %v14 = icmp eq i32 %v13, %a2
  br i1 %v14, label %b2, label %b4, !llvm.loop !5
}

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length128b,+hvxv60" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.unroll.disable"}
