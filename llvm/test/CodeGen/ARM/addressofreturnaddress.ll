; RUN: llc -mtriple=thumbv7-pc-windows-msvc < %s | FileCheck %s --check-prefixes=CHECK,WIN
; RUN: llc -mtriple=thumbv7-linux-gnueabihf < %s | FileCheck %s --check-prefixes=CHECK,ELF

; llvm.addressofreturnaddress is the address of the LR slot of the frame record,
; at the frame pointer + 4.

@r = global ptr null

declare ptr @llvm.addressofreturnaddress.p0()
declare void @callee()

; A leaf saves LR only because its address is taken.
define void @leaf() noinline {
; CHECK-LABEL: leaf:
; WIN:         push.w {r11, lr}
; WIN:         mov r11, sp
; WIN:         add.w r0, r11, #4
; ELF:         push {r7, lr}
; ELF:         mov r7, sp
; ELF:         adds r0, r7, #4
  %a = call ptr @llvm.addressofreturnaddress.p0()
  store volatile ptr %a, ptr @r
  ret void
}

; With callee-saved registers up to r11, ELF Thumb2 splits the push at r7, its
; frame pointer, so that r7 and LR stay adjacent.
@v = global i32 0
define void @manycsr() noinline {
; CHECK-LABEL: manycsr:
; WIN:         push.w {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; WIN:         add.w r11, sp, #28
; WIN:         add.w r0, r11, #4
; ELF:         push {r4, r5, r6, r7, lr}
; ELF:         add r7, sp, #12
; ELF:         push.w {r8, r9, r10, r11}
; ELF:         adds r0, r7, #4
  %x0 = load volatile i32, ptr @v
  %x1 = load volatile i32, ptr @v
  %x2 = load volatile i32, ptr @v
  %x3 = load volatile i32, ptr @v
  %x4 = load volatile i32, ptr @v
  %x5 = load volatile i32, ptr @v
  %x6 = load volatile i32, ptr @v
  %x7 = load volatile i32, ptr @v
  call void @callee()
  %a = call ptr @llvm.addressofreturnaddress.p0()
  store volatile ptr %a, ptr @r
  store volatile i32 %x0, ptr @v
  store volatile i32 %x1, ptr @v
  store volatile i32 %x2, ptr @v
  store volatile i32 %x3, ptr @v
  store volatile i32 %x4, ptr @v
  store volatile i32 %x5, ptr @v
  store volatile i32 %x6, ptr @v
  store volatile i32 %x7, ptr @v
  ret void
}
