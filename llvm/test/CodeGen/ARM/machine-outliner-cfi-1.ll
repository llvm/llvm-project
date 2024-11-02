; RUN: llc --verify-machineinstrs --force-dwarf-frame-section %s -o - | FileCheck %s
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7m-unknown-unknown-eabi"

; Derived from
; volatile int a, b, c, d, e, f, g, h;
;
; int x() {
;   int r = (a + b) / (c + d) + e + f;
;   return r + 1;
; }
;
;
; int y() {
;   int r = (a + b) / (c + d) + e + f;
;   return r + 2;
; }
; Checks that CFI instruction are emitted around saves/restores of
; the LR on stack.

@a = dso_local global i32 0, align 4
@b = dso_local global i32 0, align 4
@c = dso_local global i32 0, align 4
@d = dso_local global i32 0, align 4
@e = dso_local global i32 0, align 4
@f = dso_local global i32 0, align 4

define dso_local i32 @x() local_unnamed_addr #0 {
entry:
  %0 = load volatile i32, ptr @a, align 4
  %1 = load volatile i32, ptr @b, align 4
  %add = add nsw i32 %1, %0
  %2 = load volatile i32, ptr @c, align 4
  %3 = load volatile i32, ptr @d, align 4
  %add1 = add nsw i32 %3, %2
  %div = sdiv i32 %add, %add1
  %4 = load volatile i32, ptr @e, align 4
  %5 = load volatile i32, ptr @f, align 4
  %add2 = add i32 %div, 1
  %add3 = add i32 %add2, %4
  %add4 = add i32 %add3, %5
  ret i32 %add4
}
; CHECK-LABEL: x:
; CHECK:       str lr, [sp, #-8]!
; CHECK-NEXT:  .cfi_def_cfa_offset 8
; CHECK-NEXT:  .cfi_offset lr, -8
; CHECK-NEXT:  bl  OUTLINED_FUNCTION_0
; CHECK-NEXT:  ldr lr, [sp], #8
; CHECK-NEXT:  .cfi_def_cfa_offset 0
; CHECK-NEXT:  .cfi_restore lr

define dso_local i32 @y() local_unnamed_addr #0 {
entry:
  %0 = load volatile i32, ptr @a, align 4
  %1 = load volatile i32, ptr @b, align 4
  %add = add nsw i32 %1, %0
  %2 = load volatile i32, ptr @c, align 4
  %3 = load volatile i32, ptr @d, align 4
  %add1 = add nsw i32 %3, %2
  %div = sdiv i32 %add, %add1
  %4 = load volatile i32, ptr @e, align 4
  %5 = load volatile i32, ptr @f, align 4
  %add2 = add i32 %div, 2
  %add3 = add i32 %add2, %4
  %add4 = add i32 %add3, %5
  ret i32 %add4
}
; CHECK-LABEL: y:
; CHECK:       str lr, [sp, #-8]!
; CHECK-NEXT:  .cfi_def_cfa_offset 8
; CHECK-NEXT:  .cfi_offset lr, -8
; CHECK-NEXT:  bl  OUTLINED_FUNCTION_0
; CHECK-NEXT:  ldr lr, [sp], #8
; CHECK-NEXT:  .cfi_def_cfa_offset 0
; CHECK-NEXT:  .cfi_restore lr

attributes #0 = { minsize nofree norecurse nounwind optsize }
