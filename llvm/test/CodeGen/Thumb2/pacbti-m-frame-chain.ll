; RUN: llc -filetype asm -o - %s --frame-pointer=all -mattr=+aapcs-frame-chain -mattr=+aapcs-frame-chain-leaf -force-dwarf-frame-section | FileCheck %s
target triple = "thumbv8m.main-none-none-eabi"

; int f() {
;     return 0;
; }
;
; int x(int, char *);
; int y(int n) {
; char a[n];
; return 1 + x(n, a);
; }

define hidden i32 @f() local_unnamed_addr {
entry:
    ret i32 0;
}

define hidden i32 @x(i32 noundef %n) local_unnamed_addr {
entry:
  %vla = alloca i8, i32 %n, align 1
  %call = call i32 @y(i32 noundef %n, ptr noundef nonnull %vla)
  %add = add nsw i32 %call, 1
  ret i32 %add
}

declare dso_local i32 @y(i32 noundef, ptr noundef) local_unnamed_addr

; CHECK-LABEL: f:
; CHECK:       pac     r12, lr, sp
; CHECK-NEXT:  .save   {ra_auth_code}
; CHECK-NEXT:  str     r12, [sp, #-4]!
; CHECK-NEXT:  .cfi_def_cfa_offset 4
; CHECK-NEXT:  .cfi_offset r12, -8
; CHECK-NEXT:  .save   {r11, lr}
; CHECK-NEXT:  push.w  {r11, lr}
; CHECK-NEXT:  .cfi_offset lr, -4
; CHECK-NEXT:  .cfi_offset r11, -12
; CHECK-NEXT:  .setfp  r11, sp
; CHECK-NEXT:  mov     r11, sp
; CHECK-NEXT:  .cfi_def_cfa r11, 12
; CHECK-NEXT:  movs    r0, #0
; CHECK-NEXT:  pop.w   {r11, lr}
; CHECK-NEXT:  ldr     r12, [sp], #4
; CHECK-NEXT:  aut     r12, lr, sp
; CHECK-NEXT:  bx      lr

; CHECK-LABEL: x:
; CHECK:       pac     r12, lr, sp
; CHECK-NEXT:  .save   {r4, r7, ra_auth_code}
; CHECK-NEXT:  push.w  {r4, r7, r12}
; CHECK-NEXT:  .cfi_def_cfa_offset 12
; CHECK-NEXT:  .cfi_offset r12, -8
; CHECK-NEXT:  .cfi_offset r7, -16
; CHECK-NEXT:  .cfi_offset r4, -20
; CHECK-NEXT:  .save   {r11, lr}
; CHECK-NEXT:  push.w  {r11, lr}
; CHECK-NEXT:  .cfi_offset lr, -4
; CHECK-NEXT:  .cfi_offset r11, -12
; CHECK-NEXT:  .setfp  r11, sp
; CHECK-NEXT:  mov     r11, sp
; CHECK-NEXT:  .cfi_def_cfa_register r11
; CHECK-NEXT:  .pad    #4
; CHECK-NEXT:  sub     sp, #4
; CHECK-NEXT:  adds    r1, r0, #7
; CHECK-NEXT:  bic     r1, r1, #7
; CHECK-NEXT:  sub.w   r1, sp, r1
; CHECK-NEXT:  mov     sp, r1
; CHECK-NEXT:  bl      y
; CHECK-NEXT:  sub.w   r4, r11, #8
; CHECK-NEXT:  adds    r0, #1
; CHECK-NEXT:  mov     sp, r4
; CHECK-NEXT:  pop.w   {r11, lr}
; CHECK-NEXT:  pop.w   {r4, r7, r12}
; CHECK-NEXT:  aut     r12, lr, sp
; CHECK-NEXT:  bx      lr

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 8, !"sign-return-address", i32 1}
!1 = !{i32 8, !"sign-return-address-all", i32 0}
!2 = !{i32 8, !"branch-target-enforcement", i32 0}
