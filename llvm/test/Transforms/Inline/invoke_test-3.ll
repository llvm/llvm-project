; Test that any rethrown exceptions in an inlined function are automatically
; turned into branches to the invoke destination.

; RUN: opt < %s -inline -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline)' -S | FileCheck %s
; RUN: opt < %s -passes='module-inline' -S | FileCheck %s

declare void @might_throw()

define internal i32 @callee() personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @might_throw()
      to label %cont unwind label %exc

cont:
  ret i32 0

exc:
 ; This just rethrows the exception!
  %exn = landingpad {i8*, i32}
         cleanup
  resume { i8*, i32 } %exn
}

; caller returns true if might_throw throws an exception... which gets
; propagated by callee.
define i32 @caller() personality i32 (...)* @__gxx_personality_v0 {
; CHECK-LABEL: define i32 @caller()
entry:
  %X = invoke i32 @callee()
           to label %cont unwind label %Handler
; CHECK-NOT: @callee
; CHECK: invoke void @might_throw()
; At this point we just check that the rest of the function does not 'resume'
; at any point and instead the inlined resume is threaded into normal control
; flow.
; CHECK-NOT: resume

cont:
  ret i32 %X

Handler:
; This consumes an exception thrown by might_throw
  %exn = landingpad {i8*, i32}
         cleanup
  ret i32 1
}

declare i32 @__gxx_personality_v0(...)
