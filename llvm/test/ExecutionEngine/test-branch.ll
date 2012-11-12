; RUN: %lli %s > /dev/null

; test unconditional branch
define i32 @main() {
	br label %Test
Test:		; preds = %Test, %0
	%X = icmp eq i32 0, 4		; <i1> [#uses=1]
	br i1 %X, label %Test, label %Label
Label:		; preds = %Test
	ret i32 0
}

