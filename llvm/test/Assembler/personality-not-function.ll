; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
; Test that using a global variable (defined before use) as personality is rejected.

; CHECK: expected function

@not_a_function = global i32 42

define void @test() personality ptr @not_a_function {
entry:
  invoke void @may_throw() to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lp = landingpad { ptr, i32 } cleanup
  ret void
}

declare void @may_throw()
