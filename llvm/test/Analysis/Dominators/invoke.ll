; RUN: opt -passes=verify -disable-output < %s
; This tests that we handle unreachable blocks correctly

define void @f() personality ptr @__gxx_personality_v0 {
  %v1 = invoke ptr @g()
          to label %bb1 unwind label %bb2
  invoke void @__dynamic_cast()
          to label %bb1 unwind label %bb2
bb1:
  %Hidden = getelementptr inbounds i32, ptr %v1, i64 1
  ret void
bb2:
  %lpad.loopexit80 = landingpad { ptr, i32 }
          cleanup
  ret void
}
declare i32 @__gxx_personality_v0(...)
declare void @__dynamic_cast()
declare ptr @g()
