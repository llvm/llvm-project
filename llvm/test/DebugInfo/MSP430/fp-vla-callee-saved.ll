; RUN: llc -O0 -frame-pointer=all < %s | FileCheck %s

; Check that CFI instructions are generated properly when
; a frame pointer, variable length array and callee-saved
; registers are all present at the same time

target datalayout = "e-m:e-p:16:16-i32:16-i64:16-f32:16-f64:16-a:8-n8:16-S16"
target triple = "msp430-unknown-unknown-elf"

@g = global float 0.0
@N = external global i16, align 2

define void @foo() {
; CHECK-LABEL: foo:
; CHECK: push  r4
; CHECK-NEXT: .cfi_def_cfa_offset 4
; CHECK-NEXT: .cfi_offset r4, -4
; CHECK-NEXT: mov r1, r4
; CHECK-NEXT: .cfi_def_cfa_register r4
; CHECK: push  r6
; CHECK-NEXT: push  r7
; CHECK-NEXT: push  r8
; CHECK-NEXT: push  r9
; CHECK-NEXT: push  r10
; CHECK: .cfi_offset r6, -6
; CHECK: .cfi_offset r7, -8
; CHECK: .cfi_offset r8, -10
; CHECK: .cfi_offset r9, -12
; CHECK: .cfi_offset r10, -14

  %n = load i16, ptr @N, align 2
  %vla = alloca i8, i16 %n, align 1
  %t1 = load volatile float, ptr @g
  %t2 = load volatile float, ptr @g
  %t3 = load volatile float, ptr @g
  %t4 = load volatile float, ptr @g
  %t5 = load volatile float, ptr @g
  store volatile float %t1, ptr @g
  store volatile float %t2, ptr @g
  store volatile float %t3, ptr @g
  store volatile float %t4, ptr @g
  store volatile float %t5, ptr @g
  
; CHECK: mov r4, r1
; CHECK-NEXT: sub #10, r1
; CHECK: pop r10
; CHECK-NEXT: pop r9
; CHECK-NEXT: pop r8
; CHECK-NEXT: pop r7
; CHECK-NEXT: pop r6
; CHECK: pop r4
; CHECK: .cfi_def_cfa r1, 2
; CHECK: .cfi_restore r6
; CHECK: .cfi_restore r7
; CHECK: .cfi_restore r8
; CHECK: .cfi_restore r9
; CHECK: .cfi_restore r10

  ret void
}
