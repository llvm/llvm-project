; RUN: split-file %s %t
; RUN: cat %t/body.ll %t/none.ll > %t/none-full.ll
; RUN: cat %t/body.ll %t/zero.ll > %t/zero-full.ll
; RUN: cat %t/body.ll %t/branch.ll > %t/branch-full.ll
; RUN: llc -mtriple=x86_64 %t/none-full.ll -o - | FileCheck %s --check-prefix=NOCF
; RUN: llc -mtriple=x86_64 %t/zero-full.ll -o - | FileCheck %s --check-prefix=ZERO
; RUN: llc -mtriple=x86_64 %t/branch-full.ll -o - | FileCheck %s --check-prefix=CF

; With an unreachable default, the jump table range check can be elided unless
; cf-protection-branch is enabled. Without the range check, the table branch
; could be a JOP gadget. A value of 0 means the flag is not enabled, which
; can happen when linking merges the Min flag.

; NOCF-LABEL: switch_unreachable_default:
; NOCF:       # %bb.0: # %entry
; NOCF-NEXT:    movl %edi, %eax
; NOCF-NEXT:    jmpq *.LJTI0_0(,%rax,8)

; X86 currently only checks for the presence of the flag when emitting endbr64
; and notrack, but the range check is still elided for a value of 0.
; ZERO-LABEL: switch_unreachable_default:
; ZERO:       # %bb.0: # %entry
; ZERO-NEXT:    endbr64
; ZERO-NEXT:    movl %edi, %eax
; ZERO-NEXT:    notrack jmpq *.LJTI0_0(,%rax,8)

; CF-LABEL: switch_unreachable_default:
; CF:       # %bb.0: # %entry
; CF-NEXT:    endbr64
; CF-NEXT:    cmpl $7, %edi
; CF-NEXT:    ja .LBB0_2
; CF-NEXT:  # %bb.1: # %entry
; CF-NEXT:    movl %edi, %eax
; CF-NEXT:    notrack jmpq *.LJTI0_0(,%rax,8)

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
!llvm.module.flags = !{!0}
!0 = !{i32 8, !"cf-protection-branch", i32 1}
