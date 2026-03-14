; RUN: opt -S -passes=callsite-splitting < %s | FileCheck %s
;
; Make sure that the callsite is not splitted by checking that there's only one
; call to @callee.

; CHECK-LABEL: @caller
; CHECK-LABEL: lpad
; CHECK: call void @callee
; CHECK-NOT: call void @callee

declare void @foo(ptr %p);
declare void @bar(ptr %p);
declare dso_local i32 @__gxx_personality_v0(...)

define void @caller(ptr %p) personality ptr @__gxx_personality_v0 {
entry:
  %0 = icmp eq ptr %p, null
  br i1 %0, label %bb1, label %bb2

bb1:
  invoke void @foo(ptr %p) to label %end1 unwind label %lpad

bb2:
  invoke void @bar(ptr %p) to label %end2 unwind label %lpad

lpad:
  %1 = landingpad { ptr, i32 } cleanup
  call void @callee(ptr %p)
  resume { ptr, i32 } %1

end1:
  ret void

end2:
  ret void
}

define internal void @callee(ptr %p) {
  ret void
}
