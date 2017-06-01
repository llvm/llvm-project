; Test that we don't sink landingpads
; RUN: opt -sink -S < %s | FileCheck %s

declare hidden void @g()
declare void @h()
declare i32 @__gxx_personality_v0(...)

define void @f() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @g()
          to label %invoke.cont.15 unwind label %lpad

invoke.cont.15:
  unreachable

; CHECK: lpad:
; CHECK: %0 = landingpad { i8*, i32 }
lpad:
  %0 = landingpad { i8*, i32 }
          catch i8* null
  invoke void @h()
          to label %invoke.cont unwind label %lpad.1

; CHECK: invoke.cont
; CHECK-NOT: %0 = landingpad { i8*, i32 }
invoke.cont:
  ret void

lpad.1:
  %1 = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } %1
}
