; RUN: llc -mtriple aarch64-windows -filetype asm -o - < %s | FileCheck %s
; RUN: llc -mtriple aarch64-windows -filetype asm -o - -fast-isel %s | FileCheck %s
; RUN: llc -mtriple aarch64-windows -filetype asm -o - -global-isel -global-isel-abort=0 %s | FileCheck %s

define void @func() {
; CHECK-LABEL: func:
; CHECK:      str x30, [sp, #-16]!
; CHECK-NEXT: .seh_save_reg_x x30, 16
; CHECK-NEXT: .seh_endprologue
; CHECK-NEXT: adrp x8, .refptr.weakfunc
; CHECK-NEXT: ldr x8, [x8, .refptr.weakfunc]
; CHECK-NEXT: cbz     x8, .LBB0_2
; CHECK-NEXT: ; %bb.1:
; CHECK-NEXT: blr     x8
; CHECK-NEXT: .LBB0_2:
; CHECK-NEXT: .seh_startepilogue
; CHECK-NEXT: ldr x30, [sp], #16
; CHECK-NEXT: .seh_save_reg_x x30, 16
; CHECK-NEXT: .seh_endepilogue
; CHECK-NEXT: ret

  br i1 icmp ne (void ()* @weakfunc, void ()* null), label %1, label %2

1:
  call void @weakfunc()
  br label %2

2:
  ret void
}

declare extern_weak void @weakfunc()
