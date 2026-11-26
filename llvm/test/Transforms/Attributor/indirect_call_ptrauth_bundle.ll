; RUN: opt -passes=attributor -S < %s | FileCheck %s

; Calls with a ptrauth bundle have a signed callee, so they aren't specialized
; for their !callees.

@g = global i32 0

define void @cb1() {
  store i32 1, ptr @g
  ret void
}

define void @cb2() {
  store i32 2, ptr @g
  ret void
}

define void @multi(ptr %fp) {
; CHECK-LABEL: define void @multi(
; CHECK-NEXT:    call void %fp() [ "ptrauth"(i32 0, i64 0) ], !callees
; CHECK-NEXT:    ret void
;
  call void %fp() [ "ptrauth"(i32 0, i64 0) ], !callees !0
  ret void
}

define void @single(ptr %fp) {
; CHECK-LABEL: define void @single(
; CHECK-NEXT:    call void %fp() [ "ptrauth"(i32 0, i64 0) ], !callees
; CHECK-NEXT:    ret void
;
  call void %fp() [ "ptrauth"(i32 0, i64 0) ], !callees !1
  ret void
}

; The same calls without the bundle are specialized.

define void @multi_no_bundle(ptr %fp) {
; CHECK-LABEL: define void @multi_no_bundle(
; CHECK:         icmp eq ptr %fp, @cb1
; CHECK:         call void @cb1()
; CHECK:         call void @cb2()
;
  call void %fp(), !callees !0
  ret void
}

define void @single_no_bundle(ptr %fp) {
; CHECK-LABEL: define void @single_no_bundle(
; CHECK-NEXT:    call void @cb1()
; CHECK-NEXT:    ret void
;
  call void %fp(), !callees !1
  ret void
}

!0 = !{ptr @cb1, ptr @cb2}
!1 = !{ptr @cb1}
