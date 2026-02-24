; RUN: opt < %s -passes=simplifycfg -S | FileCheck %s

; Ensure branch weight does not overflow when folding block 3 into block 0.
; This IR test is reduced from an optimization that occurs during the Jump
; Threading pass, which creates a branch prediction of 0% and 100%. This is
; reduced in SimplifyCFG into the IR shown in the
; switch-branch-weight-overflow.ll test.

define void @foo(ptr %Overflow) {
; CHECK-LABEL: define void @foo
; CHECK:         br i1 {{.*}}, label %[[BB2:.*]], label %[[BB3:.*]], !prof [[PROF0:![0-9]+]]
  %1 = extractvalue { i32, i1 } zeroinitializer, 0
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %6, label %3

3:                                                ; preds = %0
  %4 = icmp eq i32 %1, 1
  br i1 %4, label %5, label %6, !prof !0

5:                                                ; preds = %3
  store i32 0, ptr %Overflow, align 4
  br label %6

6:                                                ; preds = %5, %3, %0
  ret void
}

; Ensure that the branch weight from folding branch 0 and the default branch
; does not overflow. This IR test is reduced from the IR above.

define void @bar(ptr %Overflow) {
; CHECK-LABEL: define void @bar
; CHECK:         br i1 {{.*}}, label %[[BB2:.*]], label %[[BB3:.*]], !prof [[PROF1:![0-9]+]]
  %1 = extractvalue { i32, i1 } zeroinitializer, 0
  switch i32 %1, label %3 [
  i32 0, label %3
  i32 1, label %2
  ], !prof !1

2:                                                ; preds = %0
  store i32 0, ptr %Overflow, align 4
  br label %3

3:                                                ; preds = %0, %0, %2
  ret void
}

; CHECK: [[PROF0]] = !{!"branch_weights", i32 0, i32 -2147483648}
; CHECK: [[PROF1]] = !{!"branch_weights", i32 500, i32 -2147483648}
!0 = !{!"branch_weights", !"expected", i32 0, i32 -2147483648}
!1 = !{!"branch_weights", !"expected", i32 -2147483648, i32 -2147483648, i32 1000}
