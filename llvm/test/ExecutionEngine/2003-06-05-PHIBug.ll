; RUN: llvm-as %s -f -o %t.bc
; RUN: lli %t.bc > /dev/null

; Testcase distilled from 256.bzip2.

target datalayout = "e-p:32:32"

define i32 @main() {
entry:
	%X = add i32 1, -1		; <i32> [#uses=3]
	br label %Next
Next:		; preds = %entry
	%A = phi i32 [ %X, %entry ]		; <i32> [#uses=0]
	%B = phi i32 [ %X, %entry ]		; <i32> [#uses=0]
	%C = phi i32 [ %X, %entry ]		; <i32> [#uses=1]
	ret i32 %C
}

