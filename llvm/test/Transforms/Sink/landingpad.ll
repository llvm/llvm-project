; Test that we don't sink landingpads
; RUN: opt -passes=sink -S < %s | FileCheck %s

declare hidden void @g()
declare void @h()
declare i32 @__gxx_personality_v0(...)

define void @f() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @g()
          to label %invoke.cont.15 unwind label %lpad

invoke.cont.15:
  unreachable

; CHECK: lpad:
; CHECK: %0 = landingpad { ptr, i32 }
lpad:
  %0 = landingpad { ptr, i32 }
          catch ptr null
  invoke void @h()
          to label %invoke.cont unwind label %lpad.1

; CHECK: invoke.cont
; CHECK-NOT: %0 = landingpad { ptr, i32 }
invoke.cont:
  ret void

lpad.1:
  %1 = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } %1
}
