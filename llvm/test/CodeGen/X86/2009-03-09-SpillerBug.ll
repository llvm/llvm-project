; RUN: llc < %s -mtriple=i386-pc-linux-gnu
; PR3706

define void @__mulxc3(x86_fp80 %b) nounwind {
entry:
	%call = call x86_fp80 @y(x86_fp80* null, x86_fp80* null)		; <x86_fp80> [#uses=0]
	%cmp = fcmp ord x86_fp80 %b, 0xK00000000000000000000		; <i1> [#uses=1]
	%sub = fsub x86_fp80 %b, %b		; <x86_fp80> [#uses=1]
	%cmp7 = fcmp uno x86_fp80 %sub, 0xK00000000000000000000		; <i1> [#uses=1]
	%and12 = and i1 %cmp7, %cmp		; <i1> [#uses=1]
	%and = zext i1 %and12 to i32		; <i32> [#uses=1]
	%conv9 = sitofp i32 %and to x86_fp80		; <x86_fp80> [#uses=1]
	store x86_fp80 %conv9, x86_fp80* null
	store x86_fp80 %b, x86_fp80* null
	ret void
}

declare x86_fp80 @y(x86_fp80*, x86_fp80*)
