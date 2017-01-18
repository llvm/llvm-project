; RUN: opt < %s -basicaa -licm -S | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop(licm)' -S %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Make sure we don't hoist the store out of the loop; %a would
; have the wrong value if f() unwinds

define void @test1(i32* nocapture noalias %a, i1 zeroext %y) uwtable {
entry:
  br label %for.body

for.body:
  %i.03 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %0 = load i32, i32* %a, align 4
  %add = add nsw i32 %0, 1
  store i32 %add, i32* %a, align 4
  br i1 %y, label %if.then, label %for.inc

; CHECK: define void @test1
; CHECK: load i32, i32*
; CHECK-NEXT: add
; CHECK-NEXT: store i32

if.then:
  tail call void @f()
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %i.03, 1
  %exitcond = icmp eq i32 %inc, 10000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

; We can hoist the store out of the loop here; if f() unwinds,
; the lifetime of %a ends.

define void @test2(i1 zeroext %y) uwtable {
entry:
  %a = alloca i32
  br label %for.body

for.body:
  %i.03 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %0 = load i32, i32* %a, align 4
  %add = add nsw i32 %0, 1
  store i32 %add, i32* %a, align 4
  br i1 %y, label %if.then, label %for.inc

if.then:
  tail call void @f()
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %i.03, 1
  %exitcond = icmp eq i32 %inc, 10000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void

; CHECK: define void @test2
; CHECK: store i32
; CHECK-NEXT: ret void
  ret void
}

declare void @f() uwtable
