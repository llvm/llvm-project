; RUN: opt -S -passes=loop-rotate -verify-memoryssa < %s | FileCheck %s
; RUN: opt -S -passes='require<target-ir>,require<assumptions>,loop(loop-rotate)' < %s | FileCheck %s
; RUN: opt -S -passes='require<target-ir>,require<assumptions>,loop-mssa(loop-rotate)' -verify-memoryssa  < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; PR5319 - The "arrayidx" gep should be hoisted, not duplicated.  We should
; end up with one phi node.
define void @test1() nounwind ssp {
; CHECK-LABEL: @test1(
entry:
  %array = alloca [20 x i32], align 16
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %i.0, 100
  br i1 %cmp, label %for.body, label %for.end

; CHECK: for.body:
; CHECK-NEXT: phi i32 [ 0
; CHECK-NEXT: store i32 0

for.body:                                         ; preds = %for.cond
  store i32 0, ptr %array, align 16
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %arrayidx.lcssa = phi ptr [ %array, %for.cond ]
  call void @g(ptr %arrayidx.lcssa) nounwind
  ret void
}

declare void @g(ptr)

; CHECK-LABEL: @test2(
define void @test2() nounwind ssp {
entry:
  %array = alloca [20 x i32], align 16
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %i.0, 100
; CHECK: call void @f
; CHECK-NOT: call void @f
  call void @f() noduplicate 
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %inc = add nsw i32 %i.0, 1
  call void @h()
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
; CHECK: }
}

declare void @f() noduplicate
declare void @h()
