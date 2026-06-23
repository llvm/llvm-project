; RUN: opt -passes=globalopt -S < %s | FileCheck %s

; GlobalOpt changing calling convention of a noipa function.
; If noipa is working, GlobalOpt should NOT change the CC of @foo,
; because it requires analyzing callers/callees interprocedurally.

; CHECK: define internal i32 @foo(
define internal i32 @foo(i32 %x) noipa noinline {
  ret i32 %x
}

define i32 @bar() {
  %r = call i32 @foo(i32 5)
  ret i32 %r
}
