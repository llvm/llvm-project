; RUN: opt -S -passes=cross-dso-cfi < %s | FileCheck --check-prefix=RISCV64 %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-linux-gnu"

define signext i8 @f() !type !0 !type !1 {
entry:
  ret i8 1
}

!llvm.module.flags = !{!2, !3}

!0 = !{i64 0, !"_ZTSFcvE"}
!1 = !{i64 0, i64 111}
!2 = !{i32 4, !"Cross-DSO CFI", i32 1}
!3 = !{i32 1, !"target-abi", !"lp64d"}

; RISCV64: define void @__cfi_check({{.*}} #[[A:.*]] align 4096
; RISCV64: attributes #[[A]] = { {{.*}}"target-features"="+d"
