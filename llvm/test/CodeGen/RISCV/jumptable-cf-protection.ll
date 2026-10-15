; RUN: split-file %s %t
; RUN: cat %t/body.ll %t/none.ll > %t/none-full.ll
; RUN: cat %t/body.ll %t/zero.ll > %t/zero-full.ll
; RUN: cat %t/body.ll %t/branch.ll > %t/branch-full.ll
; RUN: llc -mtriple=riscv64 %t/none-full.ll -o - | FileCheck %s --check-prefix=NOCF
; RUN: llc -mtriple=riscv64 %t/zero-full.ll -o - | FileCheck %s --check-prefix=NOCF
; RUN: llc -mtriple=riscv64 %t/branch-full.ll -o - | FileCheck %s --check-prefix=CF

; With an unreachable default, the jump table range check can be elided unless
; cf-protection-branch is enabled. Without the range check, the table branch
; could be a JOP gadget. A value of 0 means the flag is not enabled, which
; can happen when linking merges the Min flag.

; NOCF-LABEL: switch_unreachable_default:
; NOCF:       # %bb.0: # %entry
; NOCF-NEXT:    lui a1, %hi(.LJTI0_0)
; NOCF-NEXT:    addi a1, a1, %lo(.LJTI0_0)
; NOCF-NEXT:    slli a0, a0, 2
; NOCF-NEXT:    add a0, a1, a0
; NOCF-NEXT:    lw a0, 0(a0)
; NOCF-NEXT:    jr a0

; CF-LABEL: switch_unreachable_default:
; CF:       # %bb.0: # %entry
; CF-NEXT:    lpad 0
; CF-NEXT:    li a1, 7
; CF-NEXT:    bltu a1, a0, .LBB0_10
; CF-NEXT:  # %bb.1: # %entry
; CF-NEXT:    slli a0, a0, 2
; CF-NEXT:    lui a1, %hi(.LJTI0_0)
; CF-NEXT:    addi a1, a1, %lo(.LJTI0_0)
; CF-NEXT:    add a0, a1, a0
; CF-NEXT:    lw t2, 0(a0)
; CF-NEXT:    jr t2

;--- body.ll
define i32 @switch_unreachable_default(i32 signext %x) {
entry:
  switch i32 %x, label %def [
    i32 0, label %b0
    i32 1, label %b1
    i32 2, label %b2
    i32 3, label %b3
    i32 4, label %b4
    i32 5, label %b5
    i32 6, label %b6
    i32 7, label %b7
  ]
def:
  unreachable
b0:
  ret i32 10
b1:
  ret i32 21
b2:
  ret i32 32
b3:
  ret i32 43
b4:
  ret i32 54
b5:
  ret i32 65
b6:
  ret i32 76
b7:
  ret i32 87
}
;--- none.ll
;--- zero.ll
!llvm.module.flags = !{!0}
!0 = !{i32 8, !"cf-protection-branch", i32 0}
;--- branch.ll
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"unlabeled"}
