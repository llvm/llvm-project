; RUN: opt -passes=irce -verify-loop-info -S %s | FileCheck %s
; RUN: opt -passes='require<branch-prob>,irce' -verify-loop-info -S %s | FileCheck %s

; The branch enters the loop on the false edge of an OR of failing checks.
; Both comparisons should be false in the constrained main loop.
define void @or_of_failing_checks(ptr %arr, i32 %n, i32 %len) {
; CHECK-LABEL: define void @or_of_failing_checks(
; CHECK: loop:
; CHECK: %bad = or i1 false, false
; CHECK: br i1 %bad.not, label %in.bounds, label %out.of.bounds
entry:
  %has.iterations = icmp sgt i32 %n, 0
  br i1 %has.iterations, label %loop, label %exit

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %below = icmp slt i32 %idx, 0
  %above = icmp sge i32 %idx, %len
  %bad = or i1 %below, %above
  br i1 %bad, label %out.of.bounds, label %in.bounds, !prof !0

in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %idx.next = add nsw i32 %idx, 1
  %continue = icmp slt i32 %idx.next, %n
  br i1 %continue, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; A negated failing comparison is also a passing range check.
define void @not_of_failing_check(ptr %arr, i32 %n, i32 %len) {
; CHECK-LABEL: define void @not_of_failing_check(
; CHECK: loop:
; CHECK: %safe = xor i1 false, true
; CHECK: br i1 %safe, label %in.bounds, label %out.of.bounds
entry:
  %has.iterations = icmp sgt i32 %n, 0
  br i1 %has.iterations, label %loop, label %exit

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %above = icmp uge i32 %idx, %len
  %safe = xor i1 %above, true
  br i1 %safe, label %in.bounds, label %out.of.bounds, !prof !1

in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %idx.next = add nsw i32 %idx, 1
  %continue = icmp slt i32 %idx.next, %n
  br i1 %continue, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; Logical OR can be represented by a select. Its second condition is operand 2.
define void @select_of_failing_checks(ptr %arr, i32 %n, i32 %len) {
; CHECK-LABEL: define void @select_of_failing_checks(
; CHECK: loop:
; CHECK: %bad = select i1 false, i1 true, i1 false
entry:
  %has.iterations = icmp sgt i32 %n, 0
  br i1 %has.iterations, label %loop, label %exit

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %below = icmp slt i32 %idx, 0
  %above = icmp sge i32 %idx, %len
  %bad = select i1 %below, i1 true, i1 %above
  br i1 %bad, label %out.of.bounds, label %in.bounds, !prof !0

in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %idx.next = add nsw i32 %idx, 1
  %continue = icmp slt i32 %idx.next, %n
  br i1 %continue, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; The all-ones operand of a NOT can appear first in an XOR.
define void @commuted_not(ptr %arr, i32 %n, i32 %len) {
; CHECK-LABEL: define void @commuted_not(
; CHECK: loop:
; CHECK: %safe = xor i1 true, false
entry:
  %has.iterations = icmp sgt i32 %n, 0
  br i1 %has.iterations, label %loop, label %exit

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %above = icmp uge i32 %idx, %len
  %safe = xor i1 true, %above
  br i1 %safe, label %in.bounds, label %out.of.bounds, !prof !1

in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %idx.next = add nsw i32 %idx, 1
  %continue = icmp slt i32 %idx.next, %n
  br i1 %continue, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; An OR of passing checks does not imply that either check is true.
define void @or_of_passing_checks(ptr %arr, i32 %n, i32 %len) {
; CHECK-LABEL: define void @or_of_passing_checks(
; CHECK: loop:
; CHECK: %safe = or i1 %below, %above
entry:
  %has.iterations = icmp sgt i32 %n, 0
  br i1 %has.iterations, label %loop, label %exit

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %below = icmp slt i32 %idx, %len
  %above = icmp slt i32 %idx, %n
  %safe = or i1 %below, %above
  br i1 %safe, label %in.bounds, label %out.of.bounds, !prof !1

in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %idx.next = add nsw i32 %idx, 1
  %continue = icmp slt i32 %idx.next, %n
  br i1 %continue, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

!0 = !{!"branch_weights", i32 4, i32 64}
!1 = !{!"branch_weights", i32 64, i32 4}
