; RUN: llc -mtriple aarch64-unknown-windows-msvc -filetype asm -o - %s | FileCheck %s
; RUN: llc -mtriple aarch64-unknown-windows-msvc -filetype obj -o - %s | llvm-readobj -u - | FileCheck %s -check-prefix CHECK-UNWIND

declare dso_local void @g(ptr noundef)
define dso_local preserve_mostcc void @f(ptr noundef %p) #0 {
entry:
  %p.addr = alloca ptr, align 8
  store ptr %p, ptr %p.addr, align 8
  %0 = load ptr, ptr %p.addr, align 8
  call void @g(ptr noundef %0)
  ret void
}

attributes #0 = { nounwind uwtable(sync) }

; CHECK: str x30, [sp, #16]
; CHECK-NEXT: .seh_save_reg x30, 16
; CHECK: str x9, [sp, #24]
; CHECK-NEXT: .seh_save_any_reg x9, 24
; CHECK: stp x10, x11, [sp, #32
; CHECK-NEXT: .seh_save_any_reg_p x10, 32
; CHECK: stp x12, x13, [sp, #48]
; CHECK-NEXT: .seh_save_any_reg_p x12, 48
; CHECK: stp x14, x15, [sp, #64]
; CHECK-NEXT: .seh_save_any_reg_p x14, 64
; CHECK: .seh_endprologue

; CHECK: .seh_startepilogue
; CHECK: ldp x14, x15, [sp, #64]
; CHECK-NEXT: .seh_save_any_reg_p x14, 64
; CHECK: ldp x12, x13, [sp, #48]
; CHECK-NEXT: .seh_save_any_reg_p x12, 48
; CHECK: ldp x10, x11, [sp, #32
; CHECK-NEXT: .seh_save_any_reg_p x10, 32
; CHECK: ldr x9, [sp, #24]
; CHECK-NEXT: .seh_save_any_reg x9, 24
; CHECK: ldr x30, [sp, #16]
; CHECK-NEXT: .seh_save_reg x30, 16

; CHECK: .seh_endepilogue

; CHECK-UNWIND:  Prologue [
; CHECK-UNWIND:    0xe74e04            ; stp x14, x15, [sp, #64]
; CHECK-UNWIND:    0xe74c03            ; stp x12, x13, [sp, #48]
; CHECK-UNWIND:    0xe74a02            ; stp x10, x11, [sp, #32]
; CHECK-UNWIND:    0xe70903            ; str x9, [sp, #24]
; CHECK-UNWIND:    0xd2c2              ; str x30, [sp, #16]
; CHECK-UNWIND:    0x05                ; sub sp, #80
; CHECK-UNWIND:    0xe4                ; end
; CHECK-UNWIND:  ]
