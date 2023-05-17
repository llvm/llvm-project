; RUN: llc -march=hexagon -hexagon-initial-cfg-cleanup=0 < %s | FileCheck %s

; Check for successful compilation.
; CHECK: vmem

target triple = "hexagon"

; Function Attrs: nounwind
define dso_local void @f0(ptr %a0, ptr %a1) local_unnamed_addr #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ %v18, %b1 ], [ 0, %b0 ]
  %v1 = getelementptr inbounds i8, ptr %a1, i32 %v0
  %v2 = load <64 x i8>, ptr undef, align 1, !tbaa !0, !alias.scope !3
  %v3 = add <64 x i8> zeroinitializer, %v2
  %v4 = getelementptr inbounds i8, ptr %a0, i32 undef
  %v5 = getelementptr inbounds i8, ptr %a0, i32 undef
  %v6 = getelementptr inbounds i8, ptr %a0, i32 undef
  store i8 0, ptr undef, align 1, !tbaa !0, !alias.scope !6, !noalias !8
  %v7 = extractelement <64 x i8> %v3, i32 12
  store i8 %v7, ptr undef, align 1, !tbaa !0, !alias.scope !6, !noalias !8
  store i8 0, ptr undef, align 1, !tbaa !0, !alias.scope !6, !noalias !8
  store i8 undef, ptr %v4, align 1, !tbaa !0, !alias.scope !6, !noalias !8
  store i8 0, ptr %v5, align 1, !tbaa !0, !alias.scope !6, !noalias !8
  store i8 0, ptr undef, align 1, !tbaa !0, !alias.scope !6, !noalias !8
  %v8 = extractelement <64 x i8> %v3, i32 36
  store i8 %v8, ptr undef, align 1, !tbaa !0, !alias.scope !6, !noalias !8
  store i8 0, ptr undef, align 1, !tbaa !0, !alias.scope !6, !noalias !8
  %v9 = extractelement <64 x i8> %v3, i32 38
  store i8 %v9, ptr undef, align 1, !tbaa !0, !alias.scope !6, !noalias !8
  store i8 0, ptr undef, align 1, !tbaa !0, !alias.scope !6, !noalias !8
  %v10 = extractelement <64 x i8> %v3, i32 41
  store i8 %v10, ptr %v6, align 1, !tbaa !0, !alias.scope !6, !noalias !8
  store i8 0, ptr undef, align 1, !tbaa !0, !alias.scope !6, !noalias !8
  store i8 undef, ptr undef, align 1, !tbaa !0, !alias.scope !6, !noalias !8
  %v11 = extractelement <64 x i8> %v3, i32 55
  store i8 %v11, ptr undef, align 1, !tbaa !0, !alias.scope !6, !noalias !8
  store i8 0, ptr undef, align 1, !tbaa !0, !alias.scope !6, !noalias !8
  %v12 = extractelement <64 x i8> %v3, i32 58
  store i8 %v12, ptr undef, align 1, !tbaa !0, !alias.scope !6, !noalias !8
  store i8 0, ptr undef, align 1, !tbaa !0, !alias.scope !6, !noalias !8
  %v14 = load <64 x i8>, ptr %v1, align 1, !tbaa !0, !alias.scope !3
  %v15 = add <64 x i8> zeroinitializer, %v14
  %v16 = getelementptr inbounds i8, ptr %a0, i32 undef
  %v17 = extractelement <64 x i8> %v15, i32 23
  store i8 %v17, ptr %v16, align 1, !tbaa !0, !alias.scope !6, !noalias !8
  %v18 = add i32 %v0, 64
  br label %b1
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length64b,+hvxv60" }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!4}
!4 = distinct !{!4, !5}
!5 = distinct !{!5, !"LVerDomain"}
!6 = !{!7}
!7 = distinct !{!7, !5}
!8 = !{!4, !9, !10, !11}
!9 = distinct !{!9, !5}
!10 = distinct !{!10, !5}
!11 = distinct !{!11, !5}
