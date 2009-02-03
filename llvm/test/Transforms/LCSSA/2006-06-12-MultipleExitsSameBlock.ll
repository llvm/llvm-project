; RUN: llvm-as < %s | opt -lcssa | llvm-dis | \
; RUN:    grep {%X.1.lcssa}
; RUN: llvm-as < %s | opt -lcssa | llvm-dis | \
; RUN:    not grep {%X.1.lcssa1}

declare i1 @c1()

declare i1 @c2()

define i32 @foo() {
entry:
	br label %loop_begin
loop_begin:		; preds = %loop_body.2, %entry
	br i1 true, label %loop_body.1, label %loop_exit2
loop_body.1:		; preds = %loop_begin
	%X.1 = add i32 0, 1		; <i32> [#uses=1]
	%rel.1 = call i1 @c1( )		; <i1> [#uses=1]
	br i1 %rel.1, label %loop_exit, label %loop_body.2
loop_body.2:		; preds = %loop_body.1
	%rel.2 = call i1 @c2( )		; <i1> [#uses=1]
	br i1 %rel.2, label %loop_exit, label %loop_begin
loop_exit:		; preds = %loop_body.2, %loop_body.1
	ret i32 %X.1
loop_exit2:		; preds = %loop_begin
	ret i32 1
}

