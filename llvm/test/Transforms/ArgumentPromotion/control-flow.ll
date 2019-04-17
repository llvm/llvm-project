; RUN: opt < %s -argpromotion -S | FileCheck %s
; RUN: opt < %s -passes=argpromotion -S | FileCheck %s

; Don't promote around control flow.
define internal i32 @callee(i1 %C, i32* %P) {
; CHECK-LABEL: define internal i32 @callee(
; CHECK: i1 %C, i32* %P)
entry:
  br i1 %C, label %T, label %F

T:
  ret i32 17

F:
  %X = load i32, i32* %P
  ret i32 %X
}

define i32 @foo() {
; CHECK-LABEL: define i32 @foo(
entry:
; CHECK-NOT: load i32, i32* null
  %X = call i32 @callee(i1 true, i32* null)
; CHECK: call i32 @callee(i1 true, i32* null)
  ret i32 %X
}

