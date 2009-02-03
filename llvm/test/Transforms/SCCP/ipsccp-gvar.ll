; RUN: llvm-as < %s | opt -ipsccp | llvm-dis | not grep global

@G = internal global i32 undef		; <i32*> [#uses=5]

define void @foo() {
	%X = load i32* @G		; <i32> [#uses=1]
	store i32 %X, i32* @G
	ret void
}

define i32 @bar() {
	%V = load i32* @G		; <i32> [#uses=2]
	%C = icmp eq i32 %V, 17		; <i1> [#uses=1]
	br i1 %C, label %T, label %F
T:		; preds = %0
	store i32 17, i32* @G
	ret i32 %V
F:		; preds = %0
	store i32 123, i32* @G
	ret i32 0
}

