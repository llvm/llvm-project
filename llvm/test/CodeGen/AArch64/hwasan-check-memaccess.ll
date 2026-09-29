; RUN: llc < %s | FileCheck %s

target triple = "aarch64--linux-android"

define ptr @f1(ptr %x0, ptr %x1) {
  ; CHECK: f1:
  ; CHECK: str x30, [sp, #-16]!
  ; CHECK-NEXT: .cfi_def_cfa_offset 16
  ; CHECK-NEXT: .cfi_offset w30, -16
  ; CHECK-NEXT: mov x9, x0
  ; CHECK-NEXT: mov x0, x1
  ; CHECK-NEXT: bl __hwasan_check_x1_1
  ; CHECK-NEXT: ldr x30, [sp], #16
  ; CHECK-NEXT: ret
  call void @llvm.hwasan.check.memaccess(ptr %x0, ptr %x1, i32 1)
  ret ptr %x1
}

define ptr @f2(ptr %x0, ptr %x1) {
  ; CHECK: f2:
  ; CHECK: stp x30, x20, [sp, #-16]!
  ; CHECK-NEXT: .cfi_def_cfa_offset 16
  ; CHECK-NEXT: .cfi_offset w20, -8
  ; CHECK-NEXT: .cfi_offset w30, -16
  ; CHECK-NEXT: mov x20, x1
  ; CHECK-NEXT: bl __hwasan_check_x0_2_short_v2
  ; CHECK-NEXT: ldp x30, x20, [sp], #16
  ; CHECK-NEXT: ret
  call void @llvm.hwasan.check.memaccess.shortgranules(ptr %x1, ptr %x0, i32 2)
  ret ptr %x0
}

define void @f3(ptr %x0, ptr %x1) {
  ; 0x3ff0000 (kernel, match-all = 0xff)
  call void @llvm.hwasan.check.memaccess(ptr %x0, ptr %x1, i32 67043328)
  ret void
}

define void @f4(ptr %x0, ptr %x1) {
  ; 0x1000010 (access-size-index = 0, is-write = 1, match-all = 0x0)
  call void @llvm.hwasan.check.memaccess.shortgranules(ptr %x0, ptr %x1, i32 16777232)
  ret void
}

declare void @llvm.hwasan.check.memaccess(ptr, ptr, i32)
declare void @llvm.hwasan.check.memaccess.shortgranules(ptr, ptr, i32)

; CHECK:      .section .text.hot,"axG",@progbits,__hwasan_check_x0_2_short_v2,comdat
; CHECK-NEXT: .type __hwasan_check_x0_2_short_v2,@function
; CHECK-NEXT: .weak __hwasan_check_x0_2_short_v2
; CHECK-NEXT: .hidden __hwasan_check_x0_2_short_v2
; CHECK-NEXT: __hwasan_check_x0_2_short_v2:
; CHECK-NEXT: sbfx x16, x0, #4, #52
; CHECK-NEXT: ldrb w16, [x20, x16]
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
; CHECK-NEXT: sbfx x16, x1, #4, #52
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

; CHECK:      __hwasan_check_x1_67043328:
; CHECK-NEXT: sbfx x16, x1, #4, #52
; CHECK-NEXT: ldrb w16, [x9, x16]
; CHECK-NEXT: cmp x16, x1, lsr #56
; CHECK-NEXT: b.ne .Ltmp5
; CHECK-NEXT: .Ltmp6:
; CHECK-NEXT: ret
; CHECK-NEXT: .Ltmp5:
; CHECK-NEXT: lsr x17, x1, #56
; CHECK-NEXT: cmp x17, #255
; CHECK-NEXT: b.eq .Ltmp6
; CHECK-NEXT: stp x0, x1, [sp, #-256]!
; CHECK-NEXT: stp x29, x30, [sp, #232]
; CHECK-NEXT: mov x0, x1
; CHECK-NEXT: mov x1, #0
; CHECK-NEXT: b __hwasan_tag_mismatch

; CHECK:      __hwasan_check_x1_16777232_short_v2:
; CHECK-NEXT: sbfx	x16, x1, #4, #52
; CHECK-NEXT: ldrb	w16, [x20, x16]
; CHECK-NEXT: cmp	x16, x1, lsr #56
; CHECK-NEXT: b.ne	.Ltmp7
; CHECK-NEXT: .Ltmp8:
; CHECK-NEXT: ret
; CHECK-NEXT: .Ltmp7:
; CHECK-NEXT: lsr	x17, x1, #56
; CHECK-NEXT: cmp	x17, #0
; CHECK-NEXT: b.eq	.Ltmp8
; CHECK-NEXT: cmp	w16, #15
; CHECK-NEXT: b.hi	.Ltmp9
; CHECK-NEXT: and	x17, x1, #0xf
; CHECK-NEXT: cmp	w16, w17
; CHECK-NEXT: b.ls	.Ltmp9
; CHECK-NEXT: orr	x16, x1, #0xf
; CHECK-NEXT: ldrb	w16, [x16]
; CHECK-NEXT: cmp	x16, x1, lsr #56
; CHECK-NEXT: b.eq	.Ltmp8
; CHECK-NEXT: .Ltmp9:
; CHECK-NEXT: stp	x0, x1, [sp, #-256]!
; CHECK-NEXT: stp	x29, x30, [sp, #232]
; CHECK-NEXT: mov	x0, x1
; CHECK-NEXT: mov	x1, #16
; CHECK-NEXT: adrp	x16, :got:__hwasan_tag_mismatch_v2
; CHECK-NEXT: ldr	x16, [x16, :got_lo12:__hwasan_tag_mismatch_v2]
; CHECK-NEXT: br	x16
