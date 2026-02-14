; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
; Test that forward reference used as personality, then declared as global variable, fails.

; CHECK: forward reference used as personality must be a function

; Use @notfunc as personality first (forward reference)
define void @test() personality ptr @notfunc {
entry:
  invoke void @may_throw() to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lp = landingpad { ptr, i32 } cleanup
  ret void
}

; Then declare @notfunc as a global variable (not a function)
@notfunc = global i32 42

declare void @may_throw()
