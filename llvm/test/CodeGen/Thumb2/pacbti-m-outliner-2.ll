; RUN: llc --force-dwarf-frame-section %s -o - | FileCheck %s
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7m-arm-none-eabi"

; -Oz
; volatile int a, b, c, d, e, f;
;
; int x() {
;   int r = a + b + c + d + e + f;
;   return r + 1;
; }
;
; int y() {
;   int r = a + b + c + d + e + f;
;   return r + 2;
; }

@a = hidden global i32 0, align 4
@b = hidden global i32 0, align 4
@c = hidden global i32 0, align 4
@d = hidden global i32 0, align 4
@e = hidden global i32 0, align 4
@f = hidden global i32 0, align 4

define hidden i32 @x() local_unnamed_addr #0 {
entry:
  %0 = load volatile i32, i32* @a, align 4
  %1 = load volatile i32, i32* @b, align 4
  %2 = load volatile i32, i32* @c, align 4
  %3 = load volatile i32, i32* @d, align 4
  %4 = load volatile i32, i32* @e, align 4
  %5 = load volatile i32, i32* @f, align 4
  %add = add i32 %0, 1
  %add1 = add i32 %add, %1
  %add2 = add i32 %add1, %2
  %add3 = add i32 %add2, %3
  %add4 = add i32 %add3, %4
  %add5 = add i32 %add4, %5
  ret i32 %add5
}
; CHECK-LABEL: x:
; CHECK:        ldr   r{{.*}}, .LCPI0_0
; CHECK-NEXT:   mov   r[[A:[0-9]*]], lr
; CHECK-NEXT:   .cfi_register lr, r[[A]]
; CHECK-NEXT:   bl    OUTLINED_FUNCTION_0
; CHECK-NEXT:   mov   lr, r[[A]]
; CHECK-NEXT:   .cfi_restore lr
; CHECK-NEXT:   adds  r0, #1
; CHECK-NEXT:   bx    lr

define hidden i32 @y() local_unnamed_addr #0 {
entry:
  %0 = load volatile i32, i32* @a, align 4
  %1 = load volatile i32, i32* @b, align 4
  %2 = load volatile i32, i32* @c, align 4
  %3 = load volatile i32, i32* @d, align 4
  %4 = load volatile i32, i32* @e, align 4
  %5 = load volatile i32, i32* @f, align 4
  %add = add i32 %0, 2
  %add1 = add i32 %add, %1
  %add2 = add i32 %add1, %2
  %add3 = add i32 %add2, %3
  %add4 = add i32 %add3, %4
  %add5 = add i32 %add4, %5
  ret i32 %add5
}
; CHECK-LABEL: y:
; CHECK:       ldr   r{{.*}}, .LCPI1_0
; CHECK-NEXT:  mov   r[[B:[0-9]*]], lr
; CHECK-NEXT:  .cfi_register lr, r[[B]]
; CHECK-NEXT:  bl    OUTLINED_FUNCTION_0
; CHECK-NEXT:  mov   lr, r[[B]]
; CHECK-NEXT:  .cfi_restore lr
; CHECK-NEXT:  adds  r0, #2
; CHECK-NEXT:  bx    lr

; CHECK-LABEL: OUTLINED_FUNCTION_0:
; CHECK-NOT: pac
; CHECK-NOT: aut
; CHECK-NOT: r12
; CHECK:     bx lr

attributes #0 = { minsize nofree norecurse nounwind optsize}

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"branch-target-enforcement", i32 0}
!1 = !{i32 1, !"sign-return-address", i32 1}
!2 = !{i32 1, !"sign-return-address-all", i32 0}
