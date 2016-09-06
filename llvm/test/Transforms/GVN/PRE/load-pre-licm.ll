; RUN: opt -S -basicaa -gvn < %s | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin11.0.0"

@sortlist = external global [5001 x i32], align 4

define void @Bubble() nounwind noinline {
; CHECK: entry:
; CHECK-NEXT: %tmp7.pre = load i32
entry:
  br label %while.body5

; CHECK: while.body5:
; CHECK: %tmp7 = phi i32
; CHECK-NOT: %tmp7 = load i32
while.body5:
  %indvar = phi i32 [ 0, %entry ], [ %tmp6, %if.end ]
  %tmp5 = add i32 %indvar, 2
  %arrayidx9 = getelementptr [5001 x i32], [5001 x i32]* @sortlist, i32 0, i32 %tmp5
  %tmp6 = add i32 %indvar, 1
  %arrayidx = getelementptr [5001 x i32], [5001 x i32]* @sortlist, i32 0, i32 %tmp6
  %tmp7 = load i32, i32* %arrayidx, align 4
  %tmp10 = load i32, i32* %arrayidx9, align 4
  %cmp11 = icmp sgt i32 %tmp7, %tmp10
  br i1 %cmp11, label %if.then, label %if.end

; CHECK: if.then:
if.then:
  store i32 %tmp10, i32* %arrayidx, align 4
  store i32 %tmp7, i32* %arrayidx9, align 4
  br label %if.end

if.end:
  %exitcond = icmp eq i32 %tmp6, 100
  br i1 %exitcond, label %while.end.loopexit, label %while.body5

while.end.loopexit:
  ret void
}
