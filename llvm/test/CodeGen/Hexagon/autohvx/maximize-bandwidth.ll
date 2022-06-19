; RUN: opt -march=hexagon -hexagon-autohvx -passes=loop-vectorize -S < %s | FileCheck %s
; Check that the loop is vectorized with VF=32.
; CHECK: wide.load{{.*}} = load <32 x i32>
; CHECK: wide.load{{.*}} = load <32 x i16>

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define dso_local void @example10a(i16* noalias nocapture %a0, i16* noalias nocapture readonly %a1, i16* noalias nocapture readonly %a2, i32* noalias nocapture %a3, i32* noalias nocapture readonly %a4, i32* noalias nocapture readonly %a5) local_unnamed_addr #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ 0, %b0 ], [ %v13, %b1 ]
  %v1 = getelementptr inbounds i32, i32* %a4, i32 %v0
  %v2 = load i32, i32* %v1, align 4, !tbaa !1
  %v3 = getelementptr inbounds i32, i32* %a5, i32 %v0
  %v4 = load i32, i32* %v3, align 4, !tbaa !1
  %v5 = add nsw i32 %v4, %v2
  %v6 = getelementptr inbounds i32, i32* %a3, i32 %v0
  store i32 %v5, i32* %v6, align 4, !tbaa !1
  %v7 = getelementptr inbounds i16, i16* %a1, i32 %v0
  %v8 = load i16, i16* %v7, align 2, !tbaa !5
  %v9 = getelementptr inbounds i16, i16* %a2, i32 %v0
  %v10 = load i16, i16* %v9, align 2, !tbaa !5
  %v11 = add i16 %v10, %v8
  %v12 = getelementptr inbounds i16, i16* %a0, i32 %v0
  store i16 %v11, i16* %v12, align 2, !tbaa !5
  %v13 = add nuw nsw i32 %v0, 1
  %v14 = icmp eq i32 %v13, 1024
  br i1 %v14, label %b2, label %b1

b2:                                               ; preds = %b1
  ret void
}

attributes #0 = { noinline norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length64b,+hvxv60" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"short", !3, i64 0}
