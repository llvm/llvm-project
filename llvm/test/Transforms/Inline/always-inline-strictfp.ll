; RUN: opt -passes=always-inline,verify %s -S | FileCheck %s

; Check strictfp alwaysinline callee isn't inlined into non-strictfp caller.

declare void @foo()

define void @callee() alwaysinline strictfp {
entry:
  call void @foo() strictfp
  ret void
}

; CHECK: define void @caller()
; CHECK:   call void @callee()
define void @caller() {
entry:
  call void @callee()
  ret void
}
