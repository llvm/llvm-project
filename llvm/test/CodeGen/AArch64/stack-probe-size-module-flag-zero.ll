; RUN: split-file %s %t
; RUN: llvm-link -S %t/a.ll %t/b.ll -o %t/ab.ll
; RUN: FileCheck %s --check-prefix=FLAG < %t/ab.ll
; RUN: llc -mtriple=aarch64 < %t/ab.ll -verify-machineinstrs | FileCheck %s

; Linking a module with the "stack-probe-size" module flag (Min behavior) and
; one without it sets the flag to 0. Functions without a "stack-probe-size"
; attribute then use the default probe size of 4096. b.ll needs some module
; flag, otherwise the IRMover skips merging module flags entirely.

; FLAG: !{i32 8, !"stack-probe-size", i32 0}

; CHECK-LABEL: f:
; CHECK:         sub sp, sp, #1, lsl #12 // =4096
; CHECK-NEXT:    ldr xzr, [sp]
; CHECK-NEXT:    sub sp, sp, #1, lsl #12 // =4096
; CHECK-NEXT:    ldr xzr, [sp]
; CHECK-NEXT:    sub sp, sp, #1024

;--- a.ll
!llvm.module.flags = !{!0}
!0 = !{i32 8, !"stack-probe-size", i32 8192}

;--- b.ll
define void @f(ptr %out) nounwind "probe-stack"="inline-asm" {
  %v = alloca i8, i64 9216
  store ptr %v, ptr %out
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"wchar_size", i32 4}
