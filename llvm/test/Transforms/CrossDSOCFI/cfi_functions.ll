; Test that types referenced in ThinLTO-style !cfi.functions are known to __cfi_check.
; RUN: opt -S -passes=cross-dso-cfi < %s | FileCheck %s

; CHECK:      define void @__cfi_check(
; CHECK:        switch i64
; CHECK-NEXT:     i64 1234, label
; CHECK-NEXT:     i64 5678, label
; CHECK-NEXT:   ]

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

!cfi.functions = !{!0, !1}
!llvm.module.flags = !{!6}

!0 = !{!"f", i8 0, !2, !4}
!1 = !{!"g", i8 1, !3, !5}
!2 = !{i64 0, !"typeid1"}
!3 = !{i64 0, !"typeid2"}
!4 = !{i64 0, i64 1234}
!5 = !{i64 0, i64 5678}
!6 = !{i32 4, !"Cross-DSO CFI", i32 1}
