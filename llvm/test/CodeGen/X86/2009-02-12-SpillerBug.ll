; RUN: llc < %s -mtriple=i386-apple-darwin8
; PR3561

define hidden void @__mulxc3({ x86_fp80, x86_fp80 }* noalias nocapture sret({ x86_fp80, x86_fp80 }) %agg.result, x86_fp80 %a, x86_fp80 %b, x86_fp80 %c, x86_fp80 %d) nounwind {
entry:
	%0 = fmul x86_fp80 %b, %d		; <x86_fp80> [#uses=1]
	%1 = fsub x86_fp80 0xK00000000000000000000, %0		; <x86_fp80> [#uses=1]
	%2 = fadd x86_fp80 0xK00000000000000000000, 0xK00000000000000000000		; <x86_fp80> [#uses=1]
	%3 = fcmp uno x86_fp80 %1, 0xK00000000000000000000		; <i1> [#uses=1]
	%4 = fcmp uno x86_fp80 %2, 0xK00000000000000000000		; <i1> [#uses=1]
	%or.cond = and i1 %3, %4		; <i1> [#uses=1]
	br i1 %or.cond, label %bb47, label %bb71

bb47:		; preds = %entry
	%5 = fcmp uno x86_fp80 %a, 0xK00000000000000000000		; <i1> [#uses=1]
	br i1 %5, label %bb60, label %bb62

bb60:		; preds = %bb47
	%6 = tail call x86_fp80 @copysignl(x86_fp80 0xK00000000000000000000, x86_fp80 %a) nounwind readnone		; <x86_fp80> [#uses=0]
	br label %bb62

bb62:		; preds = %bb60, %bb47
	unreachable

bb71:		; preds = %entry
	ret void
}

declare x86_fp80 @copysignl(x86_fp80, x86_fp80) nounwind readnone
