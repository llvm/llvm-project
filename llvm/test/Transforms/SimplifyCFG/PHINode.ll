; -simplifycfg is not folding blocks if there is a PHI node involved.  This 
; should be fixed eventually

; RUN: opt < %s -simplifycfg -S | FileCheck %s

define i32 @main(i32 %argc) {
; <label>:0
; CHECK-NOT: br label %InlinedFunctionReturnNode
	br label %InlinedFunctionReturnNode
InlinedFunctionReturnNode:		; preds = %0
	%X = phi i32 [ 7, %0 ]		; <i32> [#uses=1]
	%Y = add i32 %X, %argc		; <i32> [#uses=1]
	ret i32 %Y
}

