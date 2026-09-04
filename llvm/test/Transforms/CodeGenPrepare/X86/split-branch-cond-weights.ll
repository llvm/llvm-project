; RUN: llc -fast-isel -stop-after=codegenprepare -mtriple=x86_64-unknown-linux-gnu -o - < %s | FileCheck %s

; Verify that CodeGenPrepare::splitBranchCondition installs the freshly
; computed (and scaled) branch weights on the two resulting branches.
;
; For an `or` of two conditions with original weights (A, B), the comment
; in splitBranchCondition prescribes:
;   Br1 weights = (A, A + 2*B)
;   Br2 weights = (A, 2*B)
;
; For an `and` of two conditions with original weights (A, B):
;   Br1 weights = (2*A + B, B)
;   Br2 weights = (2*A,     B)

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @a()
declare void @b()

; CHECK-LABEL: define void @split_or
; CHECK: br i1 %c1, label %t, label %entry.cond.split, !prof [[OR_BR1:![0-9]+]]
; CHECK: br i1 %c2, label %t, label %f, !prof [[OR_BR2:![0-9]+]]
define void @split_or(i32 %x, i32 %y) {
entry:
  %c1 = icmp eq i32 %x, 0
  %c2 = icmp eq i32 %y, 0
  %or = or i1 %c1, %c2
  br i1 %or, label %t, label %f, !prof !0
t:
  call void @a()
  ret void
f:
  call void @b()
  ret void
}

; CHECK-LABEL: define void @split_and
; CHECK: br i1 %c1, label %entry.cond.split, label %f, !prof [[AND_BR1:![0-9]+]]
; CHECK: br i1 %c2, label %t, label %f, !prof [[AND_BR2:![0-9]+]]
define void @split_and(i32 %x, i32 %y) {
entry:
  %c1 = icmp eq i32 %x, 0
  %c2 = icmp eq i32 %y, 0
  %and = and i1 %c1, %c2
  br i1 %and, label %t, label %f, !prof !0
t:
  call void @a()
  ret void
f:
  call void @b()
  ret void
}

!0 = !{!"branch_weights", i32 100, i32 7}

; Expected freshly-computed weights:
;   OR  Br1: (100, 100 + 2*7) = (100, 114)
;   OR  Br2: (100,       2*7) = (100,  14)
;   AND Br1: (2*100 + 7,   7) = (207,   7)
;   AND Br2: (2*100,       7) = (200,   7)

; CHECK-DAG: [[OR_BR1]] = !{!"branch_weights", i32 100, i32 114}
; CHECK-DAG: [[OR_BR2]] = !{!"branch_weights", i32 100, i32 14}
; CHECK-DAG: [[AND_BR1]] = !{!"branch_weights", i32 207, i32 7}
; CHECK-DAG: [[AND_BR2]] = !{!"branch_weights", i32 200, i32 7}
