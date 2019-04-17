; RUN: opt < %s -loop-reduce -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

declare void @foo(i8)

define void @not_addressing_mode(i8* %input, i64 %n) {
; CHECK-LABEL: @not_addressing_mode(
entry:
  br label %loop

loop:
; CHECK: loop:
; CHECK: %lsr.iv = phi i8* [ {{%[^,]+}}, %loop ], [ %input, %entry ]
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %i.next = add i64 %i, 1
  %j = mul i64 %i, -2
  ; (%input - 2 * %j) is not foldable. Worth another indvar.
  %p = getelementptr i8, i8* %input, i64 %j
  %v = load i8, i8* %p
; CHECK: %v = load i8, i8* %lsr.iv
  call void @foo(i8 %v)
  %exitcond = icmp slt i64 %i.next, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}
