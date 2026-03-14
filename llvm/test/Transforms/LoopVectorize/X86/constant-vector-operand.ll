; RUN: opt -mtriple=x86_64-apple-darwin -mcpu=core2 -passes=loop-vectorize,dce,instcombine -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@B = common global [1024 x i32] zeroinitializer, align 16
@A = common global [1024 x i32] zeroinitializer, align 16

; We use to not vectorize this loop because the shift was deemed to expensive.
; Now that we differentiate shift cost base on the operand value kind, we will
; vectorize this loop.
; CHECK: ashr <4 x i32>
define void @f() {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [1024 x i32], ptr @B, i64 0, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %shl = ashr i32 %0, 3
  %arrayidx2 = getelementptr inbounds [1024 x i32], ptr @A, i64 0, i64 %indvars.iv
  store i32 %shl, ptr %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
