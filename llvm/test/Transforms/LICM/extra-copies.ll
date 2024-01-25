; RUN: opt < %s -passes=licm -S | FileCheck %s
; RUN: opt -passes='require<aa>,require<target-ir>,require<scalar-evolution>,require<opt-remark-emit>,loop-mssa(licm)' < %s -S | FileCheck %s
; PR19835
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @f(i32 %x) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %storemerge4 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %mul = mul nsw i32 %x, %x
  %add2 = add nsw i32 %mul, %x
  %mul3 = add nsw i32 %add2, %mul
  %inc = add nsw i32 %storemerge4, 1
  %cmp = icmp slt i32 %inc, 100
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %a9.0.lcssa = phi i32 [ %mul3, %for.body ]
  ret i32 %a9.0.lcssa
}

; Test that there is exactly one copy of mul nsw i32 %x, %x in the exit block.
; CHECK: define i32 @f(i32 [[X:%.*]])
; CHECK: for.end:
; CHECK-NOT: mul nsw i32 [[X]], [[X]]
; CHECK: mul nsw i32 [[X]], [[X]]
; CHECK-NOT: mul nsw i32 [[X]], [[X]]
