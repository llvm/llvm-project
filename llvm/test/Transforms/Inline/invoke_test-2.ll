; Test that if an invoked function is inlined, and if that function cannot
; throw, that the dead handler is now unreachable.

; RUN: opt < %s  -passes=inline,simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | FileCheck %s

declare void @might_throw()

define internal i32 @callee() personality ptr @__gxx_personality_v0 {
enrty:
  invoke void @might_throw()
      to label %cont unwind label %exc

cont:
  ret i32 0

exc:
  %exn = landingpad {ptr, i32}
         cleanup
  ret i32 1
}

; caller returns true if might_throw throws an exception... callee cannot throw.
define i32 @caller() personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: define i32 @caller() personality ptr @__gxx_personality_v0
enrty:
  %X = invoke i32 @callee()
           to label %cont unwind label %UnreachableExceptionHandler
; CHECK-NOT: @callee
; CHECK: invoke void @might_throw()
; CHECK:     to label %[[C:.*]] unwind label %[[E:.*]]

; CHECK: [[E]]:
; CHECK:   landingpad
; CHECK:      cleanup
; CHECK:   br label %[[C]]

cont:
; CHECK: [[C]]:
  ret i32 %X
; CHECK:   %[[PHI:.*]] = phi i32
; CHECK:   ret i32 %[[PHI]]

UnreachableExceptionHandler:
; CHECK-NOT: UnreachableExceptionHandler:
  %exn = landingpad {ptr, i32}
         cleanup
  ret i32 -1
; CHECK-NOT: ret i32 -1
}
; CHECK: }

declare i32 @__gxx_personality_v0(...)
