; RUN: llc -mtriple=thumbv7-windows -mcpu=cortex-a9 -relocation-model pic -mattr=+long-calls -o - %s \
; RUN:    | FileCheck %s

declare arm_aapcs_vfpcc void @callee()

define arm_aapcs_vfpcc void @caller() nounwind {
entry:
  tail call void @callee()
  ret void
}

; CHECK-LABEL: caller
; CHECK: ldr [[REG:r[0-9]+]], [[CPI:\.LCPI[_0-9]+]]
; CHECK: bx [[REG]]
; CHECK: .p2align 2
; CHECK: [[CPI]]:
; CHECK: .long callee

