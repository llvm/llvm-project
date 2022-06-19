; RUN: llc --force-dwarf-frame-section --exception-model=arm %s -o - | FileCheck %s
; RUN: llc --filetype=obj %s --exception-model=arm -o - | llvm-readelf -s --unwind - | FileCheck %s --check-prefix=UNWIND
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

; Triple tweaked so we get 16-byte stack alignment and better test coverage.
target triple = "armv7m-none-nacl-android"

; -Oz
; volatile int a, b, c, d, e, f, g, h, i;
;
; int x() {
;   int r = (a + b) / (c + d) + e + f / g + h + i;
;   return r + 1;
; }
;
; int y() {
;   int r = (a + b) / (c + d) + e + f / g + h + i;
;   return r + 2;
; }

@a = hidden global i32 0, align 4
@b = hidden global i32 0, align 4
@c = hidden global i32 0, align 4
@d = hidden global i32 0, align 4
@e = hidden global i32 0, align 4
@f = hidden global i32 0, align 4
@g = hidden global i32 0, align 4
@h = hidden global i32 0, align 4
@i = hidden global i32 0, align 4


define hidden i32 @x() local_unnamed_addr #0 {
entry:
  %0 = load volatile i32, i32* @a, align 4
  %1 = load volatile i32, i32* @b, align 4
  %add = add nsw i32 %1, %0
  %2 = load volatile i32, i32* @c, align 4
  %3 = load volatile i32, i32* @d, align 4
  %add1 = add nsw i32 %3, %2
  %div = sdiv i32 %add, %add1
  %4 = load volatile i32, i32* @e, align 4
  %5 = load volatile i32, i32* @f, align 4
  %6 = load volatile i32, i32* @g, align 4
  %div3 = sdiv i32 %5, %6
  %7 = load volatile i32, i32* @h, align 4
  %8 = load volatile i32, i32* @i, align 4
  %add2 = add i32 %div, 1
  %add4 = add i32 %add2, %4
  %add5 = add i32 %add4, %div3
  %add6 = add i32 %add5, %7
  %add7 = add i32 %add6, %8
  ret i32 %add7
}

; CHECK-LABEL: x:
; CHECK:       ldr    r0, .LCPI0_0
; CHECK-NEXT:  pac    r12, lr, sp
; CHECK-NEXT:  .pad #8
; CHECK-NEXT:  .save  {ra_auth_code, lr}
; CHECK-NEXT:  strd   r12, lr, [sp, #-16]!
; CHECK-NEXT:  .cfi_def_cfa_offset 16
; CHECK-NEXT:  .cfi_offset lr, -12
; CHECK-NEXT:  .cfi_offset ra_auth_code, -16
; CHECK-NEXT:  bl    OUTLINED_FUNCTION_0
; CHECK-NEXT:  ldrd  r12, lr, [sp], #16
; CHECK-NEXT:  .cfi_def_cfa_offset 0
; CHECK-NEXT:  .cfi_restore lr
; CHECK-NEXT:  .cfi_undefined ra_auth_code
; CHECK-NEXT:  aut    r12, lr, sp
; CHECK-NEXT:  adds   r0, #1
; CHECK-NEXT:  bx     lr

define hidden i32 @y() local_unnamed_addr #0 {
entry:
  %0 = load volatile i32, i32* @a, align 4
  %1 = load volatile i32, i32* @b, align 4
  %add = add nsw i32 %1, %0
  %2 = load volatile i32, i32* @c, align 4
  %3 = load volatile i32, i32* @d, align 4
  %add1 = add nsw i32 %3, %2
  %div = sdiv i32 %add, %add1
  %4 = load volatile i32, i32* @e, align 4
  %5 = load volatile i32, i32* @f, align 4
  %6 = load volatile i32, i32* @g, align 4
  %div3 = sdiv i32 %5, %6
  %7 = load volatile i32, i32* @h, align 4
  %8 = load volatile i32, i32* @i, align 4
  %add2 = add i32 %div, 2
  %add4 = add i32 %add2, %4
  %add5 = add i32 %add4, %div3
  %add6 = add i32 %add5, %7
  %add7 = add i32 %add6, %8
  ret i32 %add7
}
; CHECK-LABEL: y:
; CHECK:      ldr    r0, .LCPI1_0
; CHECK-NEXT: pac    r12, lr, sp
; CHECK-NEXT:  .pad #8
; CHECK-NEXT: .save  {ra_auth_code, lr}
; CHECK-NEXT: strd   r12, lr, [sp, #-16]!
; CHECK-NEXT: .cfi_def_cfa_offset 16
; CHECK-NEXT: .cfi_offset lr, -12
; CHECK-NEXT: .cfi_offset ra_auth_code, -16
; CHECK-NEXT: bl    OUTLINED_FUNCTION_0
; CHECK-NEXT: ldrd  r12, lr, [sp], #16
; CHECK-NEXT: .cfi_def_cfa_offset 0
; CHECK-NEXT: .cfi_restore lr
; CHECK-NEXT: .cfi_undefined ra_auth_code
; CHECK-NEXT: aut   r12, lr, sp
; CHECK-NEXT: adds  r0, #2
; CHECK-NEXT: bx    lr

; CHECK-LABEL: OUTLINED_FUNCTION_0:
; CHECK-NOT: pac
; CHECK-NOT: aut
; CHECK-NOT: r12
; CHECK: bx lr

attributes #0 = { minsize nofree norecurse nounwind optsize uwtable}

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 8, !"branch-target-enforcement", i32 0}
!1 = !{i32 8, !"sign-return-address", i32 1}
!2 = !{i32 8, !"sign-return-address-all", i32 0}

; UNWIND-LABEL: FunctionAddress: 0x0
; UNWIND:       0xB4      ; pop ra_auth_code
; UNWIND-NEXT:  0x84 0x00 ; pop {lr}
; UNWIND-NEXT:  0x01      ; vsp = vsp + 8
; UNWIND-NEXT:  0xB0      ; finish

; UNWIND-LABEL: FunctionAddress: 0x20
; UNWIND:       0xB4      ; pop ra_auth_code
; UNWIND-NEXT:  0x84 0x00 ; pop {lr}
; UNWIND-NEXT:  0x01      ; vsp = vsp + 8
; UNWIND-NEXT:  0xB0      ; finish

; UNWIND-LABEL:  FunctionAddress: 0x40
; UNWIND:        Opcodes [
; UNWIND-NEXT:     0xB0      ; finish


; UNWINND-LABEL: 00000041 {{.*}} OUTLINED_FUNCTION_0
; UNWINND-LABEL: 00000001 {{.*}} x
; UNWINND-LABEL: 00000021 {{.*}} y
