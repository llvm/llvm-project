; RUN: opt -S -basicaa -licm < %s | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<opt-remark-emit>,loop(licm)' -S %s | FileCheck %s
; RUN: opt -S -basicaa -licm -enable-mssa-loop-dependency=true -verify-memoryssa < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @f() nounwind

; Don't hoist load past nounwind call.
define i32 @test1(i32* noalias nocapture readonly %a) nounwind uwtable {
; CHECK-LABEL: @test1(
entry:
  br label %for.body

; CHECK: tail call void @f()
; CHECK-NEXT: load i32
for.body:
  %i.06 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %x.05 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  tail call void @f() nounwind
  %i1 = load i32, i32* %a, align 4
  %add = add nsw i32 %i1, %x.05
  %inc = add nuw nsw i32 %i.06, 1
  %exitcond = icmp eq i32 %inc, 1000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i32 %add
}

; Don't hoist division past nounwind call.
define i32 @test2(i32 %N, i32 %c) nounwind uwtable {
; CHECK-LABEL: @test2(
entry:
  %cmp4 = icmp sgt i32 %N, 0
  br i1 %cmp4, label %for.body, label %for.cond.cleanup

; CHECK: tail call void @f()
; CHECK-NEXT: sdiv i32
for.body:
  %i.05 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  tail call void @f() nounwind
  %div = sdiv i32 5, %c
  %add = add i32 %i.05, 1
  %inc = add i32 %add, %div
  %cmp = icmp slt i32 %inc, %N
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret i32 0
}

; Hoist a non-volatile load past volatile load.
define i32 @test3(i32* noalias nocapture readonly %a, i32* %v) nounwind uwtable {
; CHECK-LABEL: @test3(
entry:
  br label %for.body

; CHECK: load i32
; CHECK: for.body:
; CHECK: load volatile i32
; CHECK-NOT: load
for.body:
  %i.06 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %x.05 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %xxx = load volatile i32, i32* %v, align 4
  %i1 = load i32, i32* %a, align 4
  %add = add nsw i32 %i1, %x.05
  %inc = add nuw nsw i32 %i.06, 1
  %exitcond = icmp eq i32 %inc, 1000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i32 %add
}

; Don't a volatile load past volatile load.
define i32 @test4(i32* noalias nocapture readonly %a, i32* %v) nounwind uwtable {
; CHECK-LABEL: @test4(
entry:
  br label %for.body

; CHECK: for.body:
; CHECK: load volatile i32
; CHECK-NEXT: load volatile i32
for.body:
  %i.06 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %x.05 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %xxx = load volatile i32, i32* %v, align 4
  %i1 = load volatile i32, i32* %a, align 4
  %add = add nsw i32 %i1, %x.05
  %inc = add nuw nsw i32 %i.06, 1
  %exitcond = icmp eq i32 %inc, 1000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i32 %add
}