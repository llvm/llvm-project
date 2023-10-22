; RUN: opt -passes="require<scalar-evolution>,print<scalar-evolution>,loop-vectorize" --verify-scev --verify-scev-strict -S < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @pr66616(
; CHECK: vector.body
define void @pr66616() {
entry:
  br label %for.body

for.cond5.preheader:                              ; preds = %for.body
  br label %while.body.i

for.body:                                         ; preds = %for.body, %entry
  %storemerge12 = phi i8 [ 0, %entry ], [ %inc, %for.body ]
  %0 = load i8, ptr null, align 1
  %conv2 = sext i8 %0 to i32
  %add3 = add i32 %conv2, 1
  %inc = add i8 %storemerge12, 1
  %conv1 = zext i8 %inc to i32
  %tobool.not = icmp eq i32 %conv1, 0
  br i1 %tobool.not, label %for.cond5.preheader, label %for.body

while.body.i:                                     ; preds = %while.body.i, %for.cond5.preheader
  %i.addr.09.i = phi i32 [ %dec.i, %while.body.i ], [ %add3, %for.cond5.preheader ]
  %incdec.ptr48.i = phi ptr [ %incdec.ptr.i, %while.body.i ], [ null, %for.cond5.preheader ]
  %dec.i = add i32 %i.addr.09.i, 1
  %incdec.ptr.i = getelementptr i8, ptr %incdec.ptr48.i, i64 1
  %tobool.not.i = icmp eq i32 %i.addr.09.i, 0
  br i1 %tobool.not.i, label %for.inc9.loopexit, label %while.body.i

for.inc9.loopexit:                                ; preds = %while.body.i
  ret void
}
