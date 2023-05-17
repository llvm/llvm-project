; RUN: opt < %s -passes=lcssa -disable-output
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; PR28608
; Check that we don't crash on this test.

define void @foo() {
entry:
  br label %bb1

bb1:
  br label %bb2

bb2:
  %x = phi i32 [ undef, %bb5 ], [ undef, %bb1 ]
  br i1 undef, label %bb3, label %bb6

bb3:
  br i1 undef, label %bb5, label %bb4

bb4:
  br label %bb6

bb5:
  br label %bb2

bb6:
  br label %bb1

exit:
  %y = add i32 0, %x
  ret void
}

