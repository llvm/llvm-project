; RUN: llc < %s -mtriple=aarch64-windows | FileCheck %s

define dso_local i32 @func(i32 %in) {
entry:
  call void asm sideeffect "", "~{x19}"()
  switch i32 %in, label %def [
    i32 0, label %lbl1
    i32 1, label %lbl2
    i32 2, label %lbl3
    i32 4, label %lbl4
  ]

def:
  ret i32 0

lbl1:
  call void asm sideeffect "", ""()
  ret i32 1

lbl2:
  ret i32 2

lbl3:
  ret i32 4

lbl4:
  ret i32 8
}

!llvm.module.flags = !{!0, !1}

!0 = !{i32 8, !"branch-target-enforcement", i32 1}
!1 = !{i32 8, !"sign-return-address", i32 1}

; CHECK-LABEL: func:
; CHECK-NEXT: .seh_proc func
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: hint #34
; CHECK-NEXT: .seh_nop
; CHECK-NEXT: str x19, [sp, #-16]!
; CHECK-NEXT: .seh_save_reg_x x19, 16
; CHECK-NEXT: .seh_endprologue

; CHECK:      .LBB0_2:
; CHECK-NEXT: hint #36
; CHECK: mov w0, #1

; CHECK:      .LBB0_3:
; CHECK-NEXT: hint #36
; CHECK-NEXT: mov w0, #2

; CHECK:      .LBB0_4:
; CHECK-NEXT: hint #36
; CHECK-NEXT: mov w0, #4

; CHECK:      .LBB0_5:
; CHECK-NEXT: hint #36
; CHECK-NEXT: mov w0, #8

; CHECK:      .seh_startepilogue
; CHECK-NEXT: ldr x19, [sp], #16
; CHECK-NEXT: .seh_save_reg_x x19, 16
; CHECK-NEXT: .seh_endepilogue
; CHECK-NEXT: ret
; CHECK-NEXT: .seh_endfunclet
