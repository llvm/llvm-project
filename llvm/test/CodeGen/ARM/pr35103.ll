; RUN: llc -O2 -mtriple arm < %s | FileCheck %s

; Function Attrs: norecurse nounwind readnone
define i32 @foo(i32 %vreg0, i32 %vreg1, i32 %vreg2, i32 %vreg3, i32 %vreg4) local_unnamed_addr {
entry:
  %conv = zext i32 %vreg2 to i64
  %conv1 = zext i32 %vreg0 to i64
  %add2 = add nuw nsw i64 %conv, %conv1
  %shr = lshr i64 %add2, 32
  %conv4 = trunc i64 %shr to i32
  %conv5 = and i64 %add2, 4294967295
  %add8 = add nuw nsw i64 %conv5, %conv1
  %shr9 = lshr i64 %add8, 32
  %conv10 = trunc i64 %shr9 to i32
  %add11 = add nuw nsw i32 %conv10, %conv4
  %conv12 = zext i32 %vreg3 to i64
  %conv14 = zext i32 %vreg1 to i64
  %add15 = add nuw nsw i64 %conv12, %conv14
  %shr16 = lshr i64 %add15, 32
  %conv19 = zext i32 %vreg4 to i64
  %add20 = add nuw nsw i64 %shr16, %conv19
  %shr22 = lshr i64 %add20, 32
  %conv23 = trunc i64 %shr22 to i32
  %add24 = add nuw nsw i32 %add11, %conv23
  ret i32 %add24

; CHECK: push	{r11, lr}
; CHECK-NEXT: adds	r2, r2, r0
; CHECK-NEXT: mov	r12, #0
; CHECK-NEXT: adc	lr, r12, #0
; CHECK-NEXT: adds	r0, r2, r0
; CHECK-NEXT: ldr	r2, [sp, #8]
; CHECK-NEXT: adc	r0, r12, #0
; CHECK-NEXT: adds	r1, r3, r1
; The interesting bit is the next instruction which looks
; like is computing a dead r1 but is actually computing a carry
; for the final adc.
; CHECK-NEXT: adcs	r1, r2, #0
; CHECK-NEXT: adc	r0, r0, lr
; CHECK-NEXT: pop	{r11, lr}
; CHECK-NEXT: mov	pc, lr

}
