; RUN: llc < %s | FileCheck %s

target triple = "aarch64--linux-android"

define i8* @f1(i8* %x0, i8* %x1) {
  ; CHECK: f1:
  ; CHECK: str x30, [sp, #-16]!
  ; CHECK-NEXT: .cfi_def_cfa_offset 16
  ; CHECK-NEXT: .cfi_offset w30, -16
  ; CHECK-NEXT: mov x9, x0
  ; CHECK-NEXT: bl __hwasan_check_x1_1
  ; CHECK-NEXT: mov x0, x1
  ; CHECK-NEXT: ldr x30, [sp], #16
  ; CHECK-NEXT: ret
  call void @llvm.hwasan.check.memaccess(i8* %x0, i8* %x1, i32 1)
  ret i8* %x1
}

define i8* @f2(i8* %x0, i8* %x1) {
  ; CHECK: f2:
  ; CHECK: str x30, [sp, #-16]!
  ; CHECK-NEXT: .cfi_def_cfa_offset 16
  ; CHECK-NEXT: .cfi_offset w30, -16
  ; CHECK-NEXT: mov x9, x1
  ; CHECK-NEXT: bl __hwasan_check_x0_2_short
  ; CHECK-NEXT: ldr x30, [sp], #16
  ; CHECK-NEXT: ret
  call void @llvm.hwasan.check.memaccess.shortgranules(i8* %x1, i8* %x0, i32 2)
  ret i8* %x0
}

declare void @llvm.hwasan.check.memaccess(i8*, i8*, i32)
declare void @llvm.hwasan.check.memaccess.shortgranules(i8*, i8*, i32)

; CHECK:      .section .text.hot,"axG",@progbits,__hwasan_check_x0_2_short,comdat
; CHECK-NEXT: .type __hwasan_check_x0_2_short,@function
; CHECK-NEXT: .weak __hwasan_check_x0_2_short
; CHECK-NEXT: .hidden __hwasan_check_x0_2_short
; CHECK-NEXT: __hwasan_check_x0_2_short:
; CHECK-NEXT: ubfx x16, x0, #4, #52
; CHECK-NEXT: ldrb w16, [x9, x16]
; CHECK-NEXT: cmp x16, x0, lsr #56
; CHECK-NEXT: b.ne .Ltmp0
; CHECK-NEXT: .Ltmp1:
; CHECK-NEXT: ret
; CHECK-NEXT: .Ltmp0:
; CHECK-NEXT: cmp w16, #15
; CHECK-NEXT: b.hi .Ltmp2
; CHECK-NEXT: and x17, x0, #0xf
; CHECK-NEXT: add x17, x17, #3
; CHECK-NEXT: cmp w16, w17
; CHECK-NEXT: b.ls .Ltmp2
; CHECK-NEXT: orr x16, x0, #0xf
; CHECK-NEXT: ldrb w16, [x16]
; CHECK-NEXT: cmp x16, x0, lsr #56
; CHECK-NEXT: b.eq .Ltmp1
; CHECK-NEXT: .Ltmp2:
; CHECK-NEXT: stp x0, x1, [sp, #-256]!
; CHECK-NEXT: stp x29, x30, [sp, #232]
; CHECK-NEXT: mov x1, #2
; CHECK-NEXT: adrp  x16, :got:__hwasan_tag_mismatch_v2
; CHECK-NEXT: ldr x16, [x16, :got_lo12:__hwasan_tag_mismatch_v2]
; CHECK-NEXT: br  x16


; CHECK:      .section .text.hot,"axG",@progbits,__hwasan_check_x1_1,comdat
; CHECK-NEXT: .type __hwasan_check_x1_1,@function
; CHECK-NEXT: .weak __hwasan_check_x1_1
; CHECK-NEXT: .hidden __hwasan_check_x1_1
; CHECK-NEXT: __hwasan_check_x1_1:
; CHECK-NEXT: ubfx x16, x1, #4, #52
; CHECK-NEXT: ldrb w16, [x9, x16]
; CHECK-NEXT: cmp x16, x1, lsr #56
; CHECK-NEXT: b.ne .Ltmp3
; CHECK-NEXT: .Ltmp4:
; CHECK-NEXT: ret
; CHECK-NEXT: .Ltmp3:
; CHECK-NEXT: stp x0, x1, [sp, #-256]!
; CHECK-NEXT: stp x29, x30, [sp, #232]
; CHECK-NEXT: mov x0, x1
; CHECK-NEXT: mov x1, #1
; CHECK-NEXT: adrp  x16, :got:__hwasan_tag_mismatch
; CHECK-NEXT: ldr x16, [x16, :got_lo12:__hwasan_tag_mismatch]
; CHECK-NEXT: br  x16
