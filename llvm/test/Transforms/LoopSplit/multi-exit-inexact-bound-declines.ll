; RUN: opt -passes='loop-simplify,lcssa,loop-split-test,verify' \
; RUN:   -loop-split-points=2 -S < %s | FileCheck %s

; Without -loop-split-allow-uncomputable-trip-count the fallback is off, so this
; non-unit-step symbolic-bound multi-exit loop (no computable trip count) is
; declined and left untouched -- the feature is strictly opt-in and additive.
; (multi-exit-inexact-bound.ll checks the same loop *is* split with the flag.)

; CHECK-LABEL: define i32 @multi_inexact(
; CHECK-NOT: ls.guard
; CHECK-NOT: itr.chk
; CHECK-NOT: .ls1

define i32 @multi_inexact(i32 %n) {
entry:
  br label %h

h:
  %iv = phi i32 [ 0, %entry ], [ %inc, %l ]
  %acc = phi i32 [ 0, %entry ], [ %an, %l ]
  %brk = icmp eq i32 %iv, 6
  br i1 %brk, label %side, label %l

l:
  %an = add i32 %acc, %iv
  %inc = add nuw nsw i32 %iv, 2
  %ec = icmp slt i32 %inc, %n
  br i1 %ec, label %h, label %exit

exit:
  %r = phi i32 [ %an, %l ]
  ret i32 %r

side:
  %rs = phi i32 [ %acc, %h ]
  ret i32 %rs
}
