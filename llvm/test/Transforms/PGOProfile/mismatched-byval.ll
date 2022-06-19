; RUN: opt -passes=pgo-icall-prom -profile-summary-hot-count=10 -S < %s -pass-remarks-output=- | FileCheck %s

; CHECK: byval mismatch

define void @a(i8* %0) !prof !0 {
  ret void
}

define void @b(void (i64*)** %v, i64* %p) !prof !1 {
; CHECK-LABEL: @b
; CHECK-NEXT: load
; CHECK-NEXT: call void {{.*}}(i64* byval(i64)
; CHECK-NEXT: ret void
;
  %a = load void (i64*)*, void (i64*)** %v
  call void %a(i64* byval(i64) %p), !prof !2
  ret void
}

!0 = !{!"function_entry_count", i64 36}
!1 = !{!"function_entry_count", i64 1}
!2 = !{!"VP", i32 0, i64 18, i64 12157170054180749580, i64 18}
