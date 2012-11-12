; RUN: llc < %s -pre-RA-sched=default
; RUN: llc < %s -pre-RA-sched=list-burr
; RUN: llc < %s -pre-RA-sched=fast
; PR859

; The top-down schedulers are excluded here because they don't yet support
; targets that use physreg defs.

declare i32 @printf(i8*, i32, float)

define i32 @testissue(i32 %i, float %x, float %y) {
	br label %bb1

bb1:		; preds = %bb1, %0
	%x1 = fmul float %x, %y		; <float> [#uses=1]
	%y1 = fmul float %y, 7.500000e-01		; <float> [#uses=1]
	%z1 = fadd float %x1, %y1		; <float> [#uses=1]
	%x2 = fmul float %x, 5.000000e-01		; <float> [#uses=1]
	%y2 = fmul float %y, 0x3FECCCCCC0000000		; <float> [#uses=1]
	%z2 = fadd float %x2, %y2		; <float> [#uses=1]
	%z3 = fadd float %z1, %z2		; <float> [#uses=1]
	%i1 = shl i32 %i, 3		; <i32> [#uses=1]
	%j1 = add i32 %i, 7		; <i32> [#uses=1]
	%m1 = add i32 %i1, %j1		; <i32> [#uses=2]
	%b = icmp sle i32 %m1, 6		; <i1> [#uses=1]
	br i1 %b, label %bb1, label %bb2

bb2:		; preds = %bb1
	%Msg = inttoptr i64 0 to i8*		; <i8*> [#uses=1]
	call i32 @printf( i8* %Msg, i32 %m1, float %z3 )		; <i32>:1 [#uses=0]
	ret i32 0
}
