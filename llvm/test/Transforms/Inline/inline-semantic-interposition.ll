; Check that @callee1 gets inlined while @callee2 is not, because of
; SemanticInterposition.

; RUN: opt < %s -inline -S | FileCheck %s

define internal i32 @callee1(i32 %A) {
  ret i32 %A
}

define i32 @callee2(i32 %A) {
  ret i32 %A
}

; CHECK-LABEL: @caller
define i32 @caller(i32 %A) {
; CHECK-NOT: call i32 @callee1(i32 %A)
  %A1 = call i32 @callee1(i32 %A)
; CHECK: %A2 = call i32 @callee2(i32 %A)
  %A2 = call i32 @callee2(i32 %A)
; CHECK: add i32 %A, %A2
  %R = add i32 %A1, %A2
  ret i32 %R
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"SemanticInterposition", i32 1}
