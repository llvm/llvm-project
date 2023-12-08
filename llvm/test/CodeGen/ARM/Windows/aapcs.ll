; RUN: llc -mtriple=thumbv7-windows-itanium -mcpu=cortex-a9 -o - %s | FileCheck %s

; AAPCS mandates an 8-byte stack alignment.  The alloca is implicitly aligned,
; and no bic is required.

declare void @callee(ptr %i)

define void @caller() {
  %i = alloca i8, align 8
  call void @callee(ptr %i)
  ret void
}

; CHECK: sub sp, #8
; CHECK-NOT: bic

