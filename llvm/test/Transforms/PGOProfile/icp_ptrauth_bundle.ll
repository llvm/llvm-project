; RUN: opt < %s -passes=pgo-icall-prom -pass-remarks-missed=pgo-icall-prom -S 2>&1 | FileCheck %s

; A call with a ptrauth bundle authenticates its callee, which is a signed
; pointer, before calling it. Check that it is not promoted: a direct call
; cannot have a ptrauth bundle, and the signed pointer does not compare equal
; to the address of the target.

; CHECK: remark: <unknown>:0:0: Cannot promote indirect call to func_a (count=1000): Call has a ptrauth bundle

define void @func_a() {
  ret void
}

; CHECK-LABEL: define void @with_ptrauth_bundle(
; CHECK-NEXT:    call void %fp() [ "ptrauth"(i32 0, i64 0) ], !prof ![[PROF:[0-9]+]]
; CHECK-NEXT:    ret void
define void @with_ptrauth_bundle(ptr %fp) {
  call void %fp() [ "ptrauth"(i32 0, i64 0) ], !prof !0
  ret void
}

; The same call without the bundle is promoted.
; CHECK-LABEL: define void @without_ptrauth_bundle(
; CHECK:         icmp eq ptr %fp, @func_a
; CHECK:       if.true.direct_targ:
; CHECK-NEXT:    call void @func_a()
define void @without_ptrauth_bundle(ptr %fp) {
  call void %fp(), !prof !0
  ret void
}

; CHECK: ![[PROF]] = !{!"VP", i32 0, i64 1000, i64 -9084018209228209911, i64 1000}
!0 = !{!"VP", i32 0, i64 1000, i64 -9084018209228209911, i64 1000}
