; RUN: opt < %s -S -passes=place-safepoints | FileCheck %s

declare void @llvm.localescape(...)

; Do we insert the entry safepoint after the localescape intrinsic?
define void @parent() gc "statepoint-example" {
; CHECK-LABEL: @parent
entry:
; CHECK-LABEL: entry
; CHECK-NEXT: alloca
; CHECK-NEXT: localescape
; CHECK-NEXT: call void @do_safepoint
  %ptr = alloca i32
  call void (...) @llvm.localescape(ptr %ptr)
  ret void
}

; This function is inlined when inserting a poll.  To avoid recursive 
; issues, make sure we don't place safepoints in it.
declare void @do_safepoint()
define void @gc.safepoint_poll() {
; CHECK-LABEL: gc.safepoint_poll
; CHECK-LABEL: entry
; CHECK-NEXT: do_safepoint
; CHECK-NEXT: ret void 
entry:
  call void @do_safepoint()
  ret void
}
