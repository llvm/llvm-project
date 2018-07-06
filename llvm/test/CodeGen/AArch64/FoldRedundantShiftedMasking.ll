; RUN: llc -march=aarch64 < %s | FileCheck %s -check-prefix=A64

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
; A64-LABEL: ror
; A64: mov	[[R1:w[0-9]]], #3855
; A64-NEXT:	and	[[R2:w[0-9]]], w0, [[R1]]
; A64-NEXT:	orr	[[R3:w[0-9]]], [[R1]], [[R1]], ror #8

define i32 @shl(i16 %a) {
entry:
	%0 = sext i16 %a to i32
  %1 = and i32 %0, 172
  %2 = shl i32 %0, 8
	%3 = and i32 %2, 44032
	%4 = or i32 %1, %3
	ret i32 %4
}
; A64-LABEL:shl:
; A64:		 		mov	w8, #172
; A64-NEXT: 	and	w8, w0, w8
; A64-NEXT: 	orr	w0, w8, w8, lsl #8

define i32 @lshr(i16 %a) {
entry:
	%0 = sext i16 %a to i32
	%1 = and i32 %0, 44032
  %2 = lshr i32 %0, 8
  %3 = and i32 %2, 172
	%4 = or i32 %1, %3
	ret i32 %4
}
; A64-LABEL:lshr:
; A64:		 		mov	w8, #44032
; A64-NEXT: 	and	w8, w0, w8
; A64-NEXT: 	orr	w0, w8, w8, lsr #8

define i32 @ashr(i16 %a) {
entry:
	%0 = sext i16 %a to i32
	%1 = and i32 %0, 44032
  %2 = ashr i32 %0, 8
  %3 = and i32 %2, 172
	%4 = or i32 %1, %3
	ret i32 %4
}
; A64-LABEL:ashr:
; A64:		 		mov	w8, #44032
; A64-NEXT: 	and	w8, w0, w8
; A64-NEXT: 	orr	w0, w8, w8, lsr #8


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
; A64-LABEL:shl_nogood:                             // @shl_nogood
; A64:		 		sxth	w8, w0
; A64-NEXT: 	mov	w9, #172
; A64-NEXT: 	and	w9, w8, w9
; A64-NEXT:	lsl	w8, w8, w9
; A64-NEXT:	mov	w10, #44032
; A64-NEXT:	and	w8, w8, w10
; A64-NEXT:	orr	w0, w9, w8
; A64-NEXT:		ret
; A64-LABEL:shl_nogood2:                            // @shl_nogood2
; A64:		 		sxth	w8, w0
; A64-NEXT: 	mov	w9, #172
; A64-NEXT: 	and	w9, w8, w9
; A64-NEXT:		and	w8, w8, w8, lsl #8
; A64-NEXT:		orr	w0, w9, w8
; A64-NEXT: 	ret
