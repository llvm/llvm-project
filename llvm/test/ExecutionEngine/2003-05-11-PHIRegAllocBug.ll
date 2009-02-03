; RUN: llvm-as %s -f -o %t.bc
; RUN: lli %t.bc > /dev/null

target datalayout = "e-p:32:32"

define i32 @main() {
entry:
	br label %endif
then:		; No predecessors!
	br label %endif
endif:		; preds = %then, %entry
	%x = phi i32 [ 4, %entry ], [ 27, %then ]		; <i32> [#uses=0]
	%result = phi i32 [ 32, %then ], [ 0, %entry ]		; <i32> [#uses=0]
	ret i32 0
}

