; RUN: opt %loadNPMPolly '-passes=polly-custom<scops>' -polly-print-scops -disable-output < %s 2>&1 | FileCheck %s

; CHECK-NOT: AssumedContext

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; Function Attrs: nounwind ssp uwtable
define void @hoge(ptr %arg, ptr %arg5, i32 %arg6) {
bb:
  br i1 undef, label %bb7, label %bb25

bb7:                                              ; preds = %bb21, %bb
  %tmp8 = phi i64 [ %tmp22, %bb21 ], [ 0, %bb ]
  %tmp9 = icmp sgt i32 %arg6, 0
  br i1 %tmp9, label %bb10, label %bb21

bb10:                                             ; preds = %bb10, %bb7
  %tmp11 = getelementptr inbounds [32 x [2 x float]], ptr %arg5, i64 %tmp8, i64 0
  %tmp12 = load i32, ptr %tmp11, align 4, !tbaa !4
  %tmp13 = getelementptr inbounds [32 x [2 x float]], ptr %arg5, i64 %tmp8, i64 0, i64 1
  %tmp15 = load i32, ptr %tmp13, align 4, !tbaa !4
  %tmp16 = getelementptr inbounds [38 x [64 x float]], ptr %arg, i64 1, i64 0, i64 %tmp8
  store i32 %tmp15, ptr %tmp16, align 4, !tbaa !4
  %tmp18 = add nuw nsw i64 0, 1
  %tmp19 = trunc i64 %tmp18 to i32
  %tmp20 = icmp ne i32 %tmp19, %arg6
  br i1 %tmp20, label %bb10, label %bb21

bb21:                                             ; preds = %bb10, %bb7
  %tmp22 = add nsw i64 %tmp8, 1
  %tmp23 = trunc i64 %tmp22 to i32
  %tmp24 = icmp ne i32 %tmp23, 64
  br i1 %tmp24, label %bb7, label %bb25

bb25:                                             ; preds = %bb21, %bb
  ret void
}

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 2, !"Dwarf Version", i32 2}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{i32 1, !"PIC Level", i32 2}
!3 = !{!"clang version 3.8.0 (trunk 251760) (llvm/trunk 251765)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"float", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
