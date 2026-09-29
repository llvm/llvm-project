;; RUN: llc %s -o -| FileCheck %s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux"

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @asan.module_ctor, ptr null }]

define dso_local i32 @f() #0 {
entry:
  ret i32 0
}
;; CHECK-LABEL: f:
;; CHECK: hint #34

declare void @__asan_init()
declare void @__asan_version_mismatch_check_v8()

define internal void @asan.module_ctor() #0 {
  call void @__asan_init()
  call void @__asan_version_mismatch_check_v8()
  ret void
}
;; CHECK-LABEL: asan.module_ctor:
;; CHECK: hint #34

attributes #0 = { noinline nounwind optnone sanitize_address uwtable "branch-target-enforcement" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"branch-target-enforcement", i32 1}
!2 = !{i32 4, !"sign-return-address", i32 0}
!3 = !{i32 4, !"sign-return-address-all", i32 0}
!4 = !{i32 4, !"sign-return-address-with-bkey", i32 0}