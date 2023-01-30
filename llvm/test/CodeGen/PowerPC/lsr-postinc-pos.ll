; RUN: llc -verify-machineinstrs < %s -print-lsr-output 2>&1 | FileCheck %s

; The icmp is a post-inc use, and the increment is in %bb11, but the
; scevgep needs to be inserted in %bb so that it is dominated by %t.

; CHECK: %t = load ptr, ptr %inp
; CHECK: %uglygep = getelementptr i8, ptr %t, i32 %lsr.iv.next
; CHECK: %c1 = icmp ult ptr %uglygep, %inp2

target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128-n32"
target triple = "powerpc-unknown-linux-gnu"

define void @foo(ptr %inp, ptr %inp2) nounwind {
entry:
  br label %bb11

bb11:
  %i = phi i32 [ 0, %entry ], [ %i.next, %bb ] ; <i32> [#uses=3]
  %ii = shl i32 %i, 2                       ; <i32> [#uses=1]
  %c0 = icmp eq i32 %i, 0                   ; <i1> [#uses=1]
  br i1 %c0, label %bb13, label %bb

bb:
  %t = load ptr, ptr %inp, align 16                ; <ptr> [#uses=1]
  %p = getelementptr i8, ptr %t, i32 %ii ; <ptr> [#uses=1]
  %c1 = icmp ult ptr %p,  %inp2          ; <i1> [#uses=1]
  %i.next = add i32 %i, 1                        ; <i32> [#uses=1]
  br i1 %c1, label %bb11, label %bb13

bb13:
  unreachable
}
