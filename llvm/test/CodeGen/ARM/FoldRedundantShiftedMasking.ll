target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv4t-arm-none-eabi"

; RUN: llc -march=arm < %s | FileCheck %s -check-prefix=ARM

define i32 @ror(i32 %a) {
entry:
  %m2 = and i32 %a, 3855
  %shl = shl i32 %a, 24
  %shr = lshr i32 %a, 8
  %or = or i32 %shl, %shr
  %m1 = and i32 %or, 251658255
  %or2 = or i32 %m1, %m2
  ret i32 %or2
}
; ARM-LABEL: ror
; ARM:	mov	[[R1:r[0-9]]], #15
; ARM-NEXT: orr	[[R2:r[0-9]]], [[R1]], #3840
; ARM-NEXT:	and	[[R3:r[0-9]]], r0, [[R1]]
; ARM-NEXT:	orr	[[R4:r[0-9]]], [[R3]], [[R3]], ror #8
; ARM-NEXT:	mov	pc, lr

define i32 @shl(i16 %a) {
entry:
	%0 = sext i16 %a to i32
  %1 = and i32 %0, 172
  %2 = shl i32 %0, 8
	%3 = and i32 %2, 44032
	%4 = or i32 %1, %3
	ret i32 %4
}
; ARM-LABEL: shl:
; ARM:		 	 and	r0, r0, #172
; ARM-NEXT:  orr	r0, r0, r0, lsl #8

define i32 @lshr(i16 %a) {
entry:
	%0 = sext i16 %a to i32
	%1 = and i32 %0, 44032
  %2 = lshr i32 %0, 8
  %3 = and i32 %2, 172
	%4 = or i32 %1, %3
	ret i32 %4
}
; ARM-LABEL: lshr:
; ARM:		 	 and	r0, r0, #44032
; ARM-NEXT:  orr	r0, r0, r0, lsr #8

define i32 @ashr(i16 %a) {
entry:
	%0 = sext i16 %a to i32
	%1 = and i32 %0, 44032
  %2 = ashr i32 %0, 8
  %3 = and i32 %2, 172
	%4 = or i32 %1, %3
	ret i32 %4
}
; ARM-LABEL: ashr:
; ARM:		 	 and	r0, r0, #44032
; ARM-NEXT:  orr	r0, r0, r0, lsr #8

define i32 @shl_nogood(i16 %a) {
entry:
	%0 = sext i16 %a to i32
  %1 = and i32 %0, 172
  %2 = shl i32 %0, %1
	%3 = and i32 %2, 44032
	%4 = or i32 %1, %3
	ret i32 %4
}

define i32 @shl_nogood2(i16 %a) {
entry:
	%0 = sext i16 %a to i32
  %1 = and i32 %0, 172
  %2 = shl i32 %0, 8
	%3 = and i32 %2, %0
	%4 = or i32 %1, %3
	ret i32 %4
}
; ARM-LABEL:shl_nogood:
; ARM:		 		lsl	r0, r0, #16
; ARM-NEXT: 	mov	r1, #172
; ARM-NEXT:	and	r1, r1, r0, asr #16
; ARM-NEXT:	asr	r0, r0, #16
; ARM-NEXT:	mov	r2, #44032
; ARM-NEXT:	and	r0, r2, r0, lsl r1
; ARM-NEXT:	orr	r0, r1, r0
; ARM-NEXT:	mov	pc, lr
; ARM-LABEL:shl_nogood2:
; ARM:		 		lsl	r0, r0, #16
; ARM-NEXT: 	mov	r1, #172
; ARM-NEXT:	asr	r2, r0, #16
; ARM-NEXT:	and	r1, r1, r0, asr #16
; ARM-NEXT:	lsl	r2, r2, #8
; ARM-NEXT:	and	r0, r2, r0, asr #16
; ARM-NEXT:	orr	r0, r1, r0
; ARM-NEXT:	mov	pc, lr
