; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-p:16:16-i32:16-i64:16-f32:16-f64:16-a:8-n8:16-S16"
target triple = "msp430-generic-generic"

; Test the r4-r10 callee-saved registers (MSP430 EABI p. 3.2.2).

@g = global float 0.0

define void @foo() {
; CHECK-LABEL: foo:
; CHECK-NOT: push	r11
; CHECK-NOT: push	r12
; CHECK-NOT: push	r13
; CHECK-NOT: push	r14
; CHECK-NOT: push	r15
; CHECK: push  r4
; CHECK: .cfi_def_cfa_offset 4
; CHECK: push  r5
; CHECK: .cfi_def_cfa_offset 6
; CHECK: push  r6
; CHECK: .cfi_def_cfa_offset 8
; CHECK: push  r7
; CHECK: .cfi_def_cfa_offset 10
; CHECK: push  r8
; CHECK: .cfi_def_cfa_offset 12
; CHECK: push  r9
; CHECK: .cfi_def_cfa_offset 14
; CHECK: push  r10
; CHECK: .cfi_def_cfa_offset 16

; CHECK: .cfi_offset r4, -4
; CHECK: .cfi_offset r5, -6
; CHECK: .cfi_offset r6, -8
; CHECK: .cfi_offset r7, -10
; CHECK: .cfi_offset r8, -12
; CHECK: .cfi_offset r9, -14
; CHECK: .cfi_offset r10, -16

  %t1 = load volatile float, float* @g
  %t2 = load volatile float, float* @g
  %t3 = load volatile float, float* @g
  %t4 = load volatile float, float* @g
  %t5 = load volatile float, float* @g
  %t6 = load volatile float, float* @g
  %t7 = load volatile float, float* @g
  store volatile float %t1, float* @g
  store volatile float %t2, float* @g
  store volatile float %t3, float* @g
  store volatile float %t4, float* @g
  store volatile float %t5, float* @g
  store volatile float %t6, float* @g
  ret void
}
