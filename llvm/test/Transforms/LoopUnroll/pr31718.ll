; RUN: opt -loop-unroll -verify-loop-lcssa -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@b = external local_unnamed_addr global i32, align 4

; CHECK-LABEL: @main
; CHECK: exit.loopexit:
; CHECK: {{.*}} = phi i32 [ %d.0, %h3 ]
; CHECK: br label %exit
; CHECK: exit.loopexit1:
; CHECK: {{.*}} = phi i32 [ %d.0, %h3.1 ]
; CHECK: br label %exit

define void @main(i1 %c) local_unnamed_addr #0 {
ph1:
  br label %h1

h1:
  %d.0 = phi i32 [ %1, %latch1 ], [ undef, %ph1 ]
  br label %ph2

ph2:
  br label %h2

h2:
  %0 = phi i32 [ 0, %ph2 ], [ %inc, %latch2 ]
  br label %h3

h3:
  br i1 %c, label %latch3, label %exit

latch3:
  br i1 false, label %exit3, label %h3

exit3:
  br label %latch2

latch2:
  %inc = add nuw nsw i32 %0, 1
  %cmp = icmp slt i32 %inc, 2
  br i1 %cmp, label %h2, label %exit2

exit2:
  br i1 %c, label %latch1, label %ph2

latch1:                 ; preds = %exit2
  %1 = load i32, i32* @b, align 4
  br label %h1

exit:
  %d.0.lcssa = phi i32 [ %d.0, %h3 ]
  ret void
}
