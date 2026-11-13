; RUN: split-file %s %t
; RUN: llvm-link -S %t/a.ll %t/b.ll -o %t/ab.ll
; RUN: FileCheck %s --check-prefix=FLAG < %t/ab.ll
; RUN: llc -mtriple=riscv64 < %t/ab.ll -verify-machineinstrs | FileCheck %s

; Linking a module with the "stack-probe-size" module flag (Min behavior) and
; one without it sets the flag to 0. Functions without a "stack-probe-size"
; attribute then use the default probe size of 4096. b.ll needs some module
; flag, otherwise the IRMover skips merging module flags entirely.

; FLAG: !{i32 8, !"stack-probe-size", i32 0}

; CHECK-LABEL: f:
; CHECK:         lui a0, 1
; CHECK-NEXT:    sub sp, sp, a0
; CHECK-NEXT:    sd zero, 0(sp)
; CHECK-NEXT:    addi sp, sp, -16

;--- a.ll
!llvm.module.flags = !{!0}
!0 = !{i32 8, !"stack-probe-size", i32 8192}

;--- b.ll
define i8 @f() nounwind "probe-stack"="inline-asm" {
  %a = alloca i8, i64 4096
  store volatile i8 3, ptr %a
  %c = load volatile i8, ptr %a
  ret i8 %c
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"wchar_size", i32 4}
