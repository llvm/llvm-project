; RUN: opt %loadPolly -polly-process-unprofitable -polly-print-scops -disable-output < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

; CHECK: Region: %for.cond62---%for.cond
; CHECK: p0: {0,+,1}<nuw><%for.cond>
; CHECK-NEXT: p1: %param1
; CHECK-NEXT: p2: %param2
; CHECK-NEXT: Arrays {

define void @f(ptr %param1) {
entry:
  br label %for.cond

for.cond:
  %hook = phi ptr [ %param1, %entry ], [ %add.ptr201, %cleanup ]
  br i1 undef, label %for.body, label %for.cond.cleanup

for.body:
  %param2 = call i32 @g()
  %add.ptr60 = getelementptr inbounds i8, ptr %hook, i32 %param2
  br label %for.cond62

for.cond62:
  %cmp64 = icmp ule ptr %add.ptr60, null
  br i1 %cmp64, label %for.cond62, label %cleanup

cleanup:
  %add.ptr201 = getelementptr inbounds i8, ptr %hook, i32 1
  br label %for.cond

for.cond.cleanup:
  ret void
}

declare i32 @g()
