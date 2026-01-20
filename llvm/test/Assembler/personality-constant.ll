; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
; Test that constant integer as personality is rejected.

; CHECK: expected function

define void @test() personality i32 42 {
entry:
  invoke void @may_throw() to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lp = landingpad { ptr, i32 } cleanup
  ret void
}

declare void @may_throw()
