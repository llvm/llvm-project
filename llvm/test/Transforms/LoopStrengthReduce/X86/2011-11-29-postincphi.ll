; RUN: llc < %s | FileCheck %s
;
; PR11431: handle a phi operand that is replaced by a postinc user.
; LSR first expands %t3 to %t2 in %phi
; LSR then expands %t2 in %phi into two decrements, one on each loop exit.

target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-f128:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

declare i1 @check() nounwind

; Check that LSR did something close to the behavior at the time of the bug.
; CHECK: @sqlite3DropTriggerPtr
; CHECK: incq %rax
; CHECK: jne
; CHECK: decq %rax
; CHECK: ret
define i64 @sqlite3DropTriggerPtr() nounwind {
bb:
  %cmp = call zeroext i1 @check()
  br label %bb1

bb1:                                              ; preds = %bb4, %bb
  %t0 = phi i64 [ 0, %bb ], [ %t3, %bb4 ]
  %t2 = phi i64 [ 1, %bb ], [ %t5, %bb4 ]
  %t3 = add nsw i64 %t0, 1
  br i1 %cmp, label %bb4, label %bb8

bb4:                                              ; preds = %bb1
  %t5 = add nsw i64 %t2, 1
  br i1 %cmp, label %bb1, label %bb8

bb8:                                              ; preds = %bb8, %bb4
  %phi = phi i64 [ %t3, %bb1 ], [ %t2, %bb4 ]
  ret i64 %phi
}
