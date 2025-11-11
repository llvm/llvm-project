; Check insertion and verification of indirect calls
; RUN: split-file %s %t
; RUN: opt -passes=prof-inject %t/inject.ll -S -o - | FileCheck %t/inject.ll
; RUN: opt -passes=prof-verify %t/verify-ok.ll -S -o - | FileCheck %t/verify-ok.ll
; RUN: not opt -passes=prof-verify %t/verify-bad.ll -S -o - 2>&1 | FileCheck %t/verify-bad.ll

;--- inject.ll
define void @foo(ptr %f) {
  call void %f()
  ret void
}

; CHECK: call void %f(), !prof !1
; CHECK: !0 = !{!"function_entry_count", i64 1000}
; CHECK: !1 = !{!"VP", i32 0, i64 30, i64 2345, i64 10, i64 5678, i64 20}

;--- verify-ok.ll
define void @foo(ptr %f) !prof !0 {
  call void %f(), !prof !1
  ret void
}
!0 = !{!"function_entry_count", i64 10}
!1 = !{!"VP", i32 0, i64 100, i64 123, i64 50, i64 456, i64 50}

; CHECK: call void %f(), !prof !1
; CHECK: !1 = !{!"VP", i32 0, i64 100, i64 123, i64 50, i64 456, i64 50}

;--- verify-bad.ll
define void @foo(ptr %f) !prof !0  {
  call void %f()
  ret void
}
!0 = !{!"function_entry_count", i64 10}
; CHECK: Profile verification failed: indirect call annotation missing
