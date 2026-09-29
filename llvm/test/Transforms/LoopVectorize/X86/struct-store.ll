; RUN: opt < %s -passes=loop-vectorize -mtriple=x86_64-unknown-linux-gnu -S

; Make sure we are not crashing on this one.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@glbl = external global [16 x { i64, i64 }], align 16

declare void @fn()

define void @test() {
entry:
  br label %loop

loop:
  %indvars.iv = phi i64 [ %indvars.iv.next, %loop ], [ 0, %entry ]
  %tmp = getelementptr inbounds [16 x { i64, i64 }], ptr @glbl, i64 0, i64 %indvars.iv
  store { i64, i64 } { i64 ptrtoint (ptr @fn to i64), i64 0 }, ptr %tmp, align 16
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, 16
  br i1 %exitcond, label %loop, label %exit

exit:
  ret void
}
