; RUN: opt -passes=inline %s | llc -mtriple=x86_64-unknown-linux-gnu -o - | FileCheck %s

; Opted-in callee inlined into unmarked caller: must generate NO call memset.
define internal void @callee_optin(ptr %p) #0 {
entry:
  call void @llvm.memset.p0.i64(ptr align 1 %p, i8 0, i64 512, i1 false)
  ret void
}

define void @caller_inlined(ptr %p) {
; CHECK-LABEL: caller_inlined:
; CHECK-NOT:    callq memset
; CHECK:        retq
entry:
  call void @callee_optin(ptr %p)
  ret void
}

; Unmarked caller with its own memset that inlines an opted-in callee:
; Both operations expand inline (loops), generating NO call memset.
define void @caller_with_own_memset(ptr %p1, ptr %p2) {
; CHECK-LABEL: caller_with_own_memset:
; CHECK-NOT:    callq memset
; CHECK:        retq
entry:
  call void @llvm.memset.p0.i64(ptr align 1 %p1, i8 0, i64 1024, i1 false)
  call void @callee_optin(ptr %p2)
  ret void
}

; Function opted-in directly: must generate NO call memset.
define void @optin_direct(ptr %p) #0 {
; CHECK-LABEL: optin_direct:
; CHECK-NOT:    callq memset
; CHECK:        retq
entry:
  call void @llvm.memset.p0.i64(ptr align 1 %p, i8 0, i64 512, i1 false)
  ret void
}

; Unmarked function: retains normal lowering and calls memset.
define void @unmarked_func(ptr %p) {
; CHECK-LABEL: unmarked_func:
; CHECK:        callq memset
; CHECK:        retq
entry:
  call void @llvm.memset.p0.i64(ptr align 1 %p, i8 0, i64 512, i1 false)
  ret void
}

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)

attributes #0 = { "inline-memset"="true" }
