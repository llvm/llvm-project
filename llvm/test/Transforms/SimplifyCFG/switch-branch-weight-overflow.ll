; RUN: opt < %s -passes=simplifycfg -S | FileCheck %s

; Ensure that the branch weight from folding branch 0 and the default branch
; does not overflow. This test is reduced from
; the branch-weight-overflow-simplifycfg.ll test.

define void @foo(ptr %Overflow) {
  %1 = extractvalue { i32, i1 } zeroinitializer, 0
  switch i32 %1, label %3 [
    i32 0, label %3
    i32 1, label %2
  ], !prof !0

2:                                                ; preds = %0
  store i32 0, ptr %Overflow, align 4
  br label %3

3:                                                ; preds = %0, %0, %2
  ret void
}

; CHECK: branch_weights{{.*}} 0, {{.*}} -2147483648
!0 = !{!"branch_weights", !"expected", i32 -2147483648, i32 -2147483648, i32 0}
