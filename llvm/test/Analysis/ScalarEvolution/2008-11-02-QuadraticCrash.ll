; RUN: opt < %s -analyze -enable-new-pm=0 -scalar-evolution
; RUN: opt < %s -disable-output -scalar-evolution
; PR1827

declare void @use(i32)

define void @foo() {
entry:
	br label %loop_1

loop_1:		; preds = %loop_1, %entry
	%a = phi i32 [ 2, %entry ], [ %b, %loop_1 ]		; <i32> [#uses=2]
	%c = phi i32 [ 5, %entry ], [ %d, %loop_1 ]		; <i32> [#uses=1]
	%b = add i32 %a, 1		; <i32> [#uses=1]
	%d = add i32 %c, %a		; <i32> [#uses=3]
	%A = icmp ult i32 %d, 50		; <i1> [#uses=1]
	br i1 %A, label %loop_1, label %endloop

endloop:		; preds = %loop_1
	call void @use(i32 %d)
	ret void
}
