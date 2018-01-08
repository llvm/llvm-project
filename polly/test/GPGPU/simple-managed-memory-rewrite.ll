; RUN: opt %loadPolly -analyze  -polly-process-unprofitable \
; RUN: -polly-scops -polly-use-llvm-names < %s |  FileCheck %s --check-prefix=SCOP

; RUN: opt %loadPolly -S  -polly-process-unprofitable -polly-acc-mincompute=0 \
; RUN: -polly-target=gpu  -polly-codegen-ppcg -polly-acc-codegen-managed-memory \
; RUN: -polly-acc-rewrite-managed-memory < %s | FileCheck %s --check-prefix=HOST-IR

; REQUIRES: pollyacc

; SCOP:      Function: f
; SCOP-NEXT: Region: %for.body---%for.end
; SCOP-NEXT: Max Loop Depth:  1
; SCOP: i32 MemRef_A[*];

; Check that we generate a constructor call for @A.toptr
; HOST-IR: @llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 0, void ()* {{.*}}, i8* bitcast (i32** @A.toptr to i8*) }]

; Check that we generate a constructor
; 4 bytes * 100 = 400
; HOST-IR: define void {{.*}}constructor() {
; HOST-IR-NEXT: entry:
; HOST-IR-NEXT:   %mem.raw = call i8* @polly_mallocManaged(i64 400)
; HOST-IR-NEXT:   %mem.typed = bitcast i8* %mem.raw to i32*
; HOST-IR-NEXT:   store i32* %mem.typed, i32** @A.toptr
; HOST-IR-NEXT:   ret void
; HOST-IR-NEXT: }

; HOST-IR-NOT: @A

source_filename = "test.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

@A = internal global [100 x i32] zeroinitializer, align 16

define void @f() {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %entry.split, %for.body
  %indvars.iv1 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* @A, i64 0, i64 %indvars.iv1
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
