; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s
; Test that value used in generic context, then as personality, then declared
; as global variable, correctly fails.

; CHECK: forward reference used as personality must be a function

; Use @val in generic pointer context first
define void @use_generic() {
  call void @take_ptr(ptr @val)
  ret void
}

; Then use @val as personality
define void @use_as_personality() personality ptr @val {
entry:
  invoke void @may_throw() to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lp = landingpad { ptr, i32 } cleanup
  ret void
}

; Finally declare @val as a global variable (not a function)
@val = global i32 42

declare void @take_ptr(ptr)
declare void @may_throw()
