; RUN: opt < %s -passes=loop-vectorize,instcombine -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7 -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @test1() {
entry:
  %alloca = alloca float, align 4
  br label %loop_exit.dim.11.critedge

loop_exit.dim.11.critedge:                        ; preds = %loop_body.dim.0
  %ptrint = ptrtoint ptr %alloca to i64
  %maskedptr = and i64 %ptrint, 4
  %maskcond = icmp eq i64 %maskedptr, 0
  br label %loop_header.dim.017.preheader

loop_header.dim.017.preheader:                    ; preds = %loop_exit.dim.016, %loop_exit.dim.11.critedge
  br label %loop_body.dim.018

loop_body.dim.018:                                ; preds = %loop_body.dim.018, %loop_header.dim.017.preheader
  %invar_address.dim.019.0135 = phi i64 [ 0, %loop_header.dim.017.preheader ], [ %0, %loop_body.dim.018 ]
  call void @llvm.assume(i1 %maskcond)
; CHECK:     call void @llvm.assume(
; CHECK-NOT: call void @llvm.assume(
  %0 = add nuw nsw i64 %invar_address.dim.019.0135, 1
  %1 = icmp eq i64 %0, 256
  br i1 %1, label %loop_header.dim.017.preheader, label %loop_body.dim.018
}

; Function Attrs: nounwind
declare void @llvm.assume(i1) #0

attributes #0 = { nounwind }
