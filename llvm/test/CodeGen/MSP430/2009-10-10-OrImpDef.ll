; RUN: llc -march=msp430 < %s
; PR4779 
define void @foo() nounwind {
entry:
	%r = alloca i8		; <i8*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	load volatile i8* %r, align 1		; <i8>:0 [#uses=1]
	or i8 %0, 1		; <i8>:1 [#uses=1]
	store volatile i8 %1, i8* %r, align 1
	br label %return

return:		; preds = %entry
	ret void
}
