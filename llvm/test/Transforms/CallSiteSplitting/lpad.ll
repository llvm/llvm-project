; RUN: opt -S -passes=callsite-splitting < %s | FileCheck %s
;
; Make sure that the callsite is not splitted by checking that there's only one
; call to @callee.

; CHECK-LABEL: @caller
; CHECK-LABEL: lpad
; CHECK: call void @callee
; CHECK-NOT: call void @callee

declare void @foo(i1* %p);
declare void @bar(i1* %p);
declare dso_local i32 @__gxx_personality_v0(...)

define void @caller(i1* %p) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %0 = icmp eq i1* %p, null
  br i1 %0, label %bb1, label %bb2

bb1:
  invoke void @foo(i1* %p) to label %end1 unwind label %lpad

bb2:
  invoke void @bar(i1* %p) to label %end2 unwind label %lpad

lpad:
  %1 = landingpad { i8*, i32 } cleanup
  call void @callee(i1* %p)
  resume { i8*, i32 } %1

end1:
  ret void

end2:
  ret void
}

define internal void @callee(i1* %p) {
  ret void
}
