; RUN: split-file %s %t
; RUN: opt -passes=prof-inject -profcheck-annotate-indirect-calls -S < %t/inject.ll | FileCheck %t/inject.ll
; RUN: opt -passes=prof-verify -profcheck-annotate-indirect-calls -S < %t/verify-ok.ll
; RUN: not opt -passes=prof-verify -profcheck-annotate-indirect-calls -S < %t/verify-bad.ll 2>&1 | FileCheck %t/verify-bad.ll

;--- inject.ll
define void @foo(ptr %f) {
  call void %f()
  ret void
}

; CHECK: call void %f(), !prof !1
; CHECK: !1 = !{!"VP", i32 0, i64 10, i64 19797, i64 2, i64 18770, i64 3}

;--- verify-ok.ll
define void @foo(ptr %f) !prof !0 {
  call void %f(), !prof !1
  ret void
}

!0 = !{!"function_entry_count", i32 62163}
!1 = !{!"VP", i32 0, i64 100, i64 62, i64 16, i64 3, i64 2}

;--- verify-bad.ll
define void @foo(ptr %f) !prof !0 {
  call void %f()
  ret void
}

!0 = !{!"function_entry_count", i64 62163}

; CHECK: Profile verification failed for function 'foo': indirect call annotation missing
