; RUN: llvm-as %s -o /dev/null
; RUN: verify-uselistorder %s
; Test that forward references to personality functions work correctly.

; Use @personality in a generic pointer context first (call argument)
define void @use_as_ptr() {
  call void @take_ptr(ptr @personality)
  ret void
}

; Then use @personality as a personality function
define void @use_as_personality() personality ptr @personality {
entry:
  invoke void @may_throw() to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lp = landingpad { ptr, i32 } cleanup
  ret void
}

; Finally declare @personality as a function
declare i32 @personality(...)

declare void @take_ptr(ptr)
declare void @may_throw()
