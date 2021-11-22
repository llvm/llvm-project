; Test that we can inline a simple function, turning the calls in it into invoke
; instructions

; RUN: opt < %s -inline -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline)' -S | FileCheck %s
; RUN: opt < %s -passes='module-inline' -S | FileCheck %s

declare void @might_throw()

define internal void @callee() {
entry:
  call void @might_throw()
  ret void
}

; caller returns true if might_throw throws an exception...
define i32 @caller() personality i32 (...)* @__gxx_personality_v0 {
; CHECK-LABEL: define i32 @caller() personality i32 (...)* @__gxx_personality_v0
entry:
  invoke void @callee()
      to label %cont unwind label %exc
; CHECK-NOT: @callee
; CHECK: invoke void @might_throw()

cont:
  ret i32 0

exc:
  %exn = landingpad {i8*, i32}
         cleanup
  ret i32 1
}

declare i32 @__gxx_personality_v0(...)
