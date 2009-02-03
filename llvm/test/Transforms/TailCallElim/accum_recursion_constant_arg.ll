; This is a more aggressive form of accumulator recursion insertion, which 
; requires noticing that X doesn't change as we perform the tailcall.  Thanks
; go out to the anonymous users of the demo script for "suggesting" 
; optimizations that should be done.  :)

; RUN: llvm-as < %s | opt -tailcallelim | llvm-dis | not grep call

define i32 @mul(i32 %x, i32 %y) {
entry:
	%tmp.1 = icmp eq i32 %y, 0		; <i1> [#uses=1]
	br i1 %tmp.1, label %return, label %endif
endif:		; preds = %entry
	%tmp.8 = add i32 %y, -1		; <i32> [#uses=1]
	%tmp.5 = call i32 @mul( i32 %x, i32 %tmp.8 )		; <i32> [#uses=1]
	%tmp.9 = add i32 %tmp.5, %x		; <i32> [#uses=1]
	ret i32 %tmp.9
return:		; preds = %entry
	ret i32 %x
}

