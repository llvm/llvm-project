; RUN: opt < %s -passes=simplifycfg -S | FileCheck %s

; Ensure branch weight does not overflow when folding block 3 into block 0.
; This IR test is reduced from an optimization that occurs during the Jump
; Threading pass, which creates a branch prediction of 0% and 100%. This is
; reduced in SimplifyCFG into the IR shown in the
; switch-branch-weight-overflow.ll test.

define void @foo(ptr %Overflow) {
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

; CHECK: branch_weights{{.*}} 0, {{.*}} -2147483648
!0 = !{!"branch_weights", !"expected", i32 0, i32 -2147483648}
