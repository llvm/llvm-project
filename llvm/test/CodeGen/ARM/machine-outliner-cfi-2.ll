; RUN: llc --verify-machineinstrs --force-dwarf-frame-section %s -o - | FileCheck %s
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7m-unknown-unknown-eabi"

; Derived from
; volatile int a, b, c, d, e, f, g, h;
;
; int x() {
;   int r = a + b + c + d + e + f;
;   return r + 1;
; }
;
;
; int y() {
;   int r = a + b + c + d + e + f;
;   return r + 2;
; }
; Check CFI instructions when LR is saved/restored to/from a register.

@a = dso_local global i32 0, align 4
@b = dso_local global i32 0, align 4
@c = dso_local global i32 0, align 4
@d = dso_local global i32 0, align 4
@e = dso_local global i32 0, align 4
@f = dso_local global i32 0, align 4

define dso_local i32 @x() local_unnamed_addr #0 {
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
; CHECK:      mov r3, lr
; CHECK-NEXT: .cfi_register lr, r3
; CHECK-NEXT: bl  OUTLINED_FUNCTION_0
; CHECK-NEXT: mov lr, r3
; CHECK-NEXT: .cfi_restore lr

define dso_local i32 @y() local_unnamed_addr #0 {
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
; CHECK:      mov r3, lr
; CHECK-NEXT: .cfi_register lr, r3
; CHECK-NEXT: bl  OUTLINED_FUNCTION_0
; CHECK-NEXT: mov lr, r3
; CHECK-NEXT: .cfi_restore lr

attributes #0 = { minsize nofree norecurse nounwind optsize  }
