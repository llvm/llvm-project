; RUN: opt -passes=inline -S < %s | FileCheck %s

; When a callee is inlined through an invoke, the calls in its body are turned
; into invokes. They should keep their metadata, e.g. !alloc_token, !memprof
; and !callsite, as they do when inlined through a call.

declare ptr @malloc(i64)
declare i32 @__gxx_personality_v0(...)

define internal ptr @callee(i64 %n) {
  %p = call ptr @malloc(i64 %n), !alloc_token !0, !memprof !1, !callsite !6
  ret ptr %p
}

define ptr @caller_call(i64 %n) {
; CHECK-LABEL: define ptr @caller_call(
; CHECK:         call ptr @malloc(i64 %n), !memprof ![[MP:[0-9]+]], !callsite ![[CS:[0-9]+]], !alloc_token ![[AT:[0-9]+]]
  %p = call ptr @callee(i64 %n), !callsite !7
  ret ptr %p
}

define ptr @caller_invoke(i64 %n) personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: define ptr @caller_invoke(
; CHECK:         invoke ptr @malloc(i64 %n)
; CHECK-NEXT:      to label %{{.*}} unwind label %lpad, !memprof ![[MP]], !callsite ![[CS]], !alloc_token ![[AT]]
entry:
  %p = invoke ptr @callee(i64 %n)
          to label %cont unwind label %lpad, !callsite !7
cont:
  ret ptr %p
lpad:
  %lp = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %lp
}

; CHECK: ![[AT]] = !{!"char", i1 false}
!0 = !{!"char", i1 false}
!1 = !{!2, !4}
!2 = !{!3, !"cold"}
!3 = !{i64 1, i64 2, i64 3}
!4 = !{!5, !"notcold"}
!5 = !{i64 1, i64 2, i64 4}
!6 = !{i64 1}
!7 = !{i64 2}
