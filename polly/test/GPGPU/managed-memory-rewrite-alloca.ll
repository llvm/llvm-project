; RUN: opt %loadPolly -analyze  -polly-process-unprofitable \
; RUN: -polly-scops -polly-use-llvm-names < %s |  FileCheck %s --check-prefix=SCOP

; RUN: opt %loadPolly -S  -polly-process-unprofitable -polly-acc-mincompute=0 \
; RUN: -polly-target=gpu  -polly-codegen-ppcg -polly-acc-codegen-managed-memory \
; RUN: -polly-acc-rewrite-managed-memory  -polly-acc-rewrite-allocas < %s | FileCheck %s --check-prefix=HOST-IR

; REQUIRES: pollyacc

; SCOP:      Function: f
; SCOP-NEXT: Region: %for.body---%for.end
; SCOP-NEXT: Max Loop Depth:  1
; SCOP: i32 MemRef_arr[*];

; Check that we generate a constructor call for @A.toptr
; HOST-IR-NOT:   %arr = alloca [100 x i32]

source_filename = "test.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"


define void @f() {
entry:
  %arr = alloca [100 x i32]
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %entry.split, %for.body
  %indvars.iv1 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* %arr, i64 0, i64 %indvars.iv1
  store i32 42, i32* %arrayidx, align 4, !tbaa !3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv1, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #0


; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #0

attributes #0 = { argmemonly nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{!"clang version 6.0.0 (http://llvm.org/git/clang.git 6660f0d30ef23b3142a6b08f9f41aad3d47c084f) (http://llvm.org/git/llvm.git 052dd78cb30f77a05dc8bb06b851402c4b6c6587)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
