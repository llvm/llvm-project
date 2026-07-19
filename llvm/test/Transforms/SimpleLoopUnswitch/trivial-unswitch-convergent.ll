; RUN: opt -passes=simple-loop-unswitch -S < %s | FileCheck %s

; simple-loop-unswitch must NOT unswitch a loop-invariant branch out of a loop
; that contains a convergent operation. The "generalized trivial unswitching"
; case (llvm/llvm-project#204934) redirects a loop-invariant branch's latch edge
; to the loop exit, which turns the branch into a loop-exit branch and lets it be
; hoisted to the preheader. When the loop body executes a convergent instruction
; on every iteration, hoisting the branch lets one path bypass the loop entirely,
; changing how many times (and for which threads) the convergent op executes.
;
; Here @conv is convergent and runs unconditionally in the loop header. If %cond
; is false the original loop still runs (executing @conv each iteration) and only
; skips %body. Unswitching the branch out makes the %cond==false path skip the
; whole loop, so @conv is never executed -- a miscompile. The branch must stay
; in the loop.

declare i32 @conv(i32) #0

define void @trivial_unswitch_convergent(i1 %cond, ptr %p) {
; CHECK-LABEL: define void @trivial_unswitch_convergent(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %header
; CHECK:       header:
; CHECK:         call i32 @conv(
; CHECK:         br i1 %cond, label %body, label %latch
; CHECK-NOT:     .split
entry:
  br label %header

header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %latch ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %latch ]
  %c = call i32 @conv(i32 %acc)
  br i1 %cond, label %body, label %latch

body:
  %add = add i32 %c, %acc
  br label %latch

latch:
  %acc.next = phi i32 [ %add, %body ], [ %acc, %header ]
  %iv.next = add i32 %iv, 1
  %done = icmp eq i32 %iv.next, 4
  br i1 %done, label %exit, label %header

exit:
  ret void
}

; Classic trivial unswitch: a convergent op runs in the header BEFORE a
; loop-invariant *exit* branch. Hoisting the branch to the preheader would skip
; the header (and the convergent op) whenever the invariant condition exits, so
; @conv would run 0 times instead of once. This case predates the latch-redirect
; form above; the branch must stay in the loop.

define void @classic_exit_branch_convergent(i1 %cond, i32 %n) {
; CHECK-LABEL: define void @classic_exit_branch_convergent(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %header
; CHECK:       header:
; CHECK:         call i32 @conv(
; CHECK:         br i1 %cond, label %exit, label %body
; CHECK-NOT:     .split
entry:
  br label %header

header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %latch ]
  %c = call i32 @conv(i32 %iv)
  br i1 %cond, label %exit, label %body

body:
  br label %latch

latch:
  %iv.next = add i32 %iv, 1
  %done = icmp eq i32 %iv.next, %n
  br i1 %done, label %exit, label %header

exit:
  ret void
}

attributes #0 = { convergent nounwind willreturn memory(none) }
