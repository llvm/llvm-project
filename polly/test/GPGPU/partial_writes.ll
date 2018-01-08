; RUN: opt %loadPolly -polly-import-jscop -polly-codegen-ppcg -S < %s \
; RUN: | FileCheck %s

; REQUIRES: pollyacc

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: polly_launchKernel

; Function Attrs: nounwind uwtable
define void @partial_writes() {
bb:
  %tmp = tail call i8* @wibble() #2
  %tmp1 = bitcast i8* %tmp to [1200 x double]*
  br label %bb2

bb2:                                              ; preds = %bb11, %bb
  %tmp3 = phi i64 [ 0, %bb ], [ %tmp12, %bb11 ]
  %tmp4 = getelementptr inbounds [1200 x double], [1200 x double]* %tmp1, i64 0, i64 %tmp3
  %tmp5 = load double, double* %tmp4, align 8, !tbaa !1
  br label %bb6

bb6:                                              ; preds = %bb6, %bb2
  %tmp7 = phi double [ undef, %bb2 ], [ undef, %bb6 ]
  %tmp8 = phi i64 [ 0, %bb2 ], [ %tmp9, %bb6 ]
  store double undef, double* %tmp4, align 8, !tbaa !1
  %tmp9 = add nuw nsw i64 %tmp8, 1
  %tmp10 = icmp eq i64 %tmp9, 900
  br i1 %tmp10, label %bb11, label %bb6

bb11:                                             ; preds = %bb6
  %tmp12 = add nuw nsw i64 %tmp3, 1
  %tmp13 = icmp eq i64 %tmp12, 1200
  br i1 %tmp13, label %bb14, label %bb2

bb14:                                             ; preds = %bb11
  ret void
}

declare i8* @wibble()


!llvm.ident = !{!0}

!0 = !{!"clang version 6.0.0 (trunk 309912) (llvm/trunk 309933)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"double", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
