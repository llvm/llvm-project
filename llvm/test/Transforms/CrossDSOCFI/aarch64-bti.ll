; RUN: opt -S -passes=cross-dso-cfi < %s | FileCheck %s

; CHECK: define void @__cfi_check({{.*}}) [[ATTR:#[0-9]+]] align 4096
; CHECK: [[ATTR]] = { "branch-target-enforcement" }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

@_ZTV1A = constant i8 0, !type !4, !type !5
@_ZTV1B = constant i8 0, !type !4, !type !5, !type !6, !type !7

define signext i8 @f11() "branch-target-enforcement" !type !0 !type !1 {
entry:
  ret i8 1
}

define signext i8 @f12() "branch-target-enforcement" !type !0 !type !1 {
entry:
  ret i8 2
}

define signext i8 @f13() "branch-target-enforcement" !type !0 !type !1 {
entry:
  ret i8 3
}

define i32 @f21() "branch-target-enforcement" !type !2 !type !3 {
entry:
  ret i32 4
}

define i32 @f22() "branch-target-enforcement" !type !2 !type !3 {
entry:
  ret i32 5
}

define weak_odr hidden void @__cfi_check_fail(ptr, ptr) "branch-target-enforcement" {
entry:
  ret void
}

!llvm.module.flags = !{!8, !9}

!0 = !{i64 0, !"_ZTSFcvE"}
!1 = !{i64 0, i64 111}
!2 = !{i64 0, !"_ZTSFivE"}
!3 = !{i64 0, i64 222}
!4 = !{i64 16, !"_ZTS1A"}
!5 = !{i64 16, i64 333}
!6 = !{i64 16, !"_ZTS1B"}
!7 = !{i64 16, i64 444}
!8 = !{i32 4, !"Cross-DSO CFI", i32 1}
!9 = !{i32 8, !"branch-target-enforcement", i32 1}
