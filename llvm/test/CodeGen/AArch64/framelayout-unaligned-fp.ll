; RUN: llc -verify-machineinstrs < %s | FileCheck %s

; The purpose of this test is to construct a scenario where an odd number
; of callee-saved GPRs as well as an odd number of callee-saved FPRs are
; used. This caused the frame pointer to be aligned to a multiple of 8
; on non-Darwin platforms, rather than a multiple of 16 as usual.

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

@a = global i64 0, align 4


define i64 @b() uwtable {
entry:
  %call = tail call i64 @d()
  %0 = alloca i8, i64 ptrtoint (ptr @d to i64), align 16
  %1 = ptrtoint ptr %0 to i64
  store i64 %1, ptr @a, align 4
  %call1 = call i64 @e()
  %conv = sitofp i64 %call1 to float
  %2 = load i64, ptr @a, align 4
  %call2 = call i64 @f(i64 %2)
  %conv3 = fptosi float %conv to i64
  ret i64 %conv3
}

; CHECK-LABEL: b:
; CHECK:       str     d8, [sp, #-32]!
; CHECK-NEXT:  .cfi_def_cfa_offset 32
; CHECK-NEXT:  stp     x29, x30, [sp, #8]
; CHECK-NEXT:  str     x19, [sp, #24]
; CHECK-NEXT:  add     x29, sp, #8

; CHECK:       sub     sp, x29, #8
; CHECK-NEXT:  .cfi_def_cfa wsp, 32
; CHECK-NEXT:  ldp     x29, x30, [sp, #8]
; CHECK-NEXT:  ldr     x19, [sp, #24]
; CHECK-NEXT:  ldr     d8, [sp], #32
; CHECK-NEXT: .cfi_def_cfa_offset 0
; CHECK-NEXT: .cfi_restore w19
; CHECK-NEXT: .cfi_restore w30
; CHECK-NEXT: .cfi_restore w29
; CHECK-NEXT: .cfi_restore b8
; CHECK-NEXT:  ret

declare i64 @d()
declare i64 @e()
declare i64 @f(i64)
