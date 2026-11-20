; RUN: split-file %s %t
; RUN: opt -passes=prof-inject -profcheck-annotate-mem-ops -S < %t/inject.ll | FileCheck %t/inject.ll
; RUN: opt -passes=prof-verify -profcheck-annotate-mem-ops -S < %t/verify-ok.ll
; RUN: not opt -passes=prof-verify -profcheck-annotate-mem-ops -S < %t/verify-bad.ll 2>&1 | FileCheck %t/verify-bad.ll

; This test fails under profcheck due to profcheck adding profile metadata.
; UNSUPPORTED: profcheck

;--- inject.ll
define void @foo(ptr %dest, ptr %src, i32 %len) {
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr %src, i32 %len, i1 false)
  ret void
}

define void @bar(ptr %dest, ptr %src) {
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr %src, i32 16, i1 false)
  ret void
}

; CHECK-LABEL: define void @foo
; CHECK: call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr %src, i32 %len, i1 false), !prof !1
; CHECK-LABEL: define void @bar
; CHECK: call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr %src, i32 16, i1 false)
; CHECK-NOT: !prof
; CHECK: !1 = !{!"VP", i32 1, i64 10, i64 19797, i64 2, i64 18770, i64 3}

;--- verify-ok.ll
define void @foo(ptr %dest, ptr %src, i32 %len) !prof !0 {
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr %src, i32 %len, i1 false), !prof !1
  ret void
}

define void @bar(ptr %dest, ptr %src) !prof !0 {
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr %src, i32 16, i1 false)
  ret void
}

!0 = !{!"function_entry_count", i64 62163}
!1 = !{!"VP", i32 1, i64 60, i64 3, i64 25, i64 2, i64 30}

;--- verify-bad.ll
define void @foo(ptr %dest, ptr %src, i32 %len) !prof !0 {
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr %src, i32 %len, i1 false)
  ret void
}

!0 = !{!"function_entry_count", i64 62163}

; CHECK: Profile verification failed for function 'foo': memory operation annotation missing
