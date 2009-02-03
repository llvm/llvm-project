; -simplifycfg is not folding blocks if there is a PHI node involved.  This 
; should be fixed eventually

; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | not grep br

define i32 @main(i32 %argc) {
; <label>:0
	br label %InlinedFunctionReturnNode
InlinedFunctionReturnNode:		; preds = %0
	%X = phi i32 [ 7, %0 ]		; <i32> [#uses=1]
	%Y = add i32 %X, %argc		; <i32> [#uses=1]
	ret i32 %Y
}

