; RUN: opt -passes='globaldce,attributor' -S %s | FileCheck %s

; A function referenced only by metadata can be deleted, leaving a raw null
; operand behind. The resulting malformed !callees attachment must be ignored
; rather than interpreted as an exhaustive empty set.

define void @caller(ptr %callee) {
; CHECK-LABEL: define void @caller(
; CHECK:         call void %callee(), !callees ![[CALLEES:[0-9]+]]
; CHECK-NEXT:    ret void
  call void %callee(), !callees !0
  ret void
}

define internal void @metadata_only_target() {
  ret void
}

; CHECK-NOT: define internal void @metadata_only_target

; A well-formed empty attachment is an exhaustive empty set. Reaching the call
; is therefore undefined, and Attributor may make the call site unreachable.
define void @empty_callees(ptr %callee) {
; CHECK-LABEL: define void @empty_callees(
; CHECK-NEXT:    unreachable
  call void %callee(), !callees !1
  ret void
}

; A declaration referenced only by metadata can also disappear while a
; different target remains live. The surviving entry is not exhaustive.
define i32 @known_target() {
  ret i32 17
}

declare i32 @external_target()

define i32 @mixed_callees(ptr %callee) {
; CHECK-LABEL: define i32 @mixed_callees(
; CHECK:         %[[R:.*]] = call i32 %callee(), !callees ![[MIXED:[0-9]+]]
; CHECK-NEXT:    ret i32 %[[R]]
  %r = call i32 %callee(), !callees !2
  ret i32 %r
}

; CHECK: ![[CALLEES]] = distinct !{null}
; CHECK: ![[MIXED]] = distinct !{ptr @known_target, null}

!0 = !{ptr @metadata_only_target}
!1 = !{}
!2 = !{ptr @known_target, ptr @external_target}
