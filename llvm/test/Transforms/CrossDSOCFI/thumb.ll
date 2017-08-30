; RUN: opt -mtriple=armv7-linux-android -S -cross-dso-cfi < %s | FileCheck --check-prefix=THUMB %s
; RUN: opt -mtriple=thumbv7-linux-android -S -cross-dso-cfi < %s | FileCheck --check-prefix=THUMB %s
; RUN: opt -mtriple=i386-linux -S -cross-dso-cfi < %s | FileCheck --check-prefix=NOTHUMB %s
; RUN: opt -mtriple=x86_64-linux -S -cross-dso-cfi < %s | FileCheck --check-prefix=NOTHUMB %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

define signext i8 @f() !type !0 !type !1 {
entry:
  ret i8 1
}

!llvm.module.flags = !{!2}

!0 = !{i64 0, !"_ZTSFcvE"}
!1 = !{i64 0, i64 111}
!2 = !{i32 4, !"Cross-DSO CFI", i32 1}

; THUMB: define void @__cfi_check({{.*}} #[[A:.*]] align 4096
; THUMB: attributes #[[A]] = { {{.*}}"target-features"="+thumb-mode"

; NOTHUMB: define void @__cfi_check({{.*}} align 4096
