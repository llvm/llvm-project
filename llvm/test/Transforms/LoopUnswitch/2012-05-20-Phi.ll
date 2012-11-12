; RUN: opt < %s -loop-unswitch -disable-output
; PR12887
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = common global i32 0, align 4
@c = common global i32 0, align 4
@b = common global i32 0, align 4

define void @func() noreturn nounwind uwtable {
entry:
  %0 = load i32* @a, align 4
  %tobool = icmp eq i32 %0, 0
  %1 = load i32* @b, align 4
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %d.0 = phi i8 [ undef, %entry ], [ %conv2, %while.body ]
  %conv = sext i8 %d.0 to i32
  %cond = select i1 %tobool, i32 0, i32 %conv
  %conv11 = zext i8 %d.0 to i32
  %add = add i32 %1, %conv11
  %conv2 = trunc i32 %add to i8
  br label %while.body
}
