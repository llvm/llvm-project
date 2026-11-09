; RUN: llc -O2 -mtriple=thumbv7-pc-windows-msvc < %s | FileCheck %s --check-prefixes=CHECK,O2W
; RUN: llc -O0 -mtriple=thumbv7-pc-windows-msvc < %s | FileCheck %s --check-prefixes=CHECK,O0W
; RUN: llc -O2 -mtriple=thumbv7-linux-gnueabihf < %s | FileCheck %s --check-prefixes=CHECK,ELF

; llvm.addressofreturnaddress is the address of the LR slot of the frame record,
; at the frame pointer + 4. A leaf function spills LR for it.

@r = global ptr null

declare ptr @llvm.addressofreturnaddress.p0()
declare void @callee()

; A leaf: LR is saved only because its address is taken.
define void @leaf() noinline {
; CHECK-LABEL: leaf:
; O2W:         push.w {r11, lr}
; O2W-NEXT:    .seh_save_regs_w {r11, lr}
; O2W-NEXT:    mov r11, sp
; O2W:         add.w r0, r11, #4
; O0W:         push.w {r11, lr}
; O0W-NEXT:    .seh_save_regs_w {r11, lr}
; O0W-NEXT:    mov r11, sp
; O0W:         mov r0, r11
; O0W-NEXT:    adds r0, #4
; ELF:         push {r7, lr}
; ELF:         mov r7, sp
; ELF:         adds r0, r7, #4
  %a = call ptr @llvm.addressofreturnaddress.p0()
  store volatile ptr %a, ptr @r
  ret void
}

; A non-leaf, which saves LR anyway.
define void @nonleaf() noinline {
; CHECK-LABEL: nonleaf:
; O2W:         push.w {r11, lr}
; O2W:         mov r11, sp
; O2W:         add.w r0, r11, #4
; O0W:         push.w {r11, lr}
; O0W:         mov r11, sp
; O0W:         mov r0, r11
; O0W-NEXT:    adds r0, #4
; ELF:         push {r7, lr}
; ELF:         mov r7, sp
; ELF:         adds r0, r7, #4
  %a = call ptr @llvm.addressofreturnaddress.p0()
  store volatile ptr %a, ptr @r
  call void @callee()
  ret void
}

; Callee-saved registers up to r11. On ELF Thumb2 the frame pointer is r7, so
; the push is split at r7 to keep r7 and LR adjacent: one push {r4-r11, lr}
; would leave r7 + 4 pointing at the saved r8. On Windows the frame pointer is
; r11, which sits right below LR in a single push.
@v = global i32 0
define void @manycsr() noinline {
; CHECK-LABEL: manycsr:
; O2W:         push.w {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; O2W:         add.w r11, sp, #28
; O2W:         add.w r0, r11, #4
; O0W:         push.w {r4, r5, r6, r7, r8, r11, lr}
; O0W:         add.w r11, sp, #20
; O0W:         mov [[R:r[0-9]+]], r11
; O0W-NEXT:    adds [[R]], #4
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
