; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
; Test that using an alias as personality is rejected.

; CHECK: expected function

declare i32 @real_personality(...)

@personality_alias = alias i32 (...), ptr @real_personality

define void @test() personality ptr @personality_alias {
entry:
  invoke void @may_throw() to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lp = landingpad { ptr, i32 } cleanup
  ret void
}

declare void @may_throw()
