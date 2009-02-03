; RUN: llvm-as < %s | opt -ipsccp | llvm-dis | \
; RUN:   grep -v {ret i32 0} | grep -v {ret i32 undef} | not grep ret

define internal i32 @bar(i32 %A) {
	%C = icmp eq i32 %A, 0		; <i1> [#uses=1]
	br i1 %C, label %T, label %F
T:		; preds = %0
	%B = call i32 @bar( i32 0 )		; <i32> [#uses=0]
	ret i32 0
F:		; preds = %0
	%C.upgrd.1 = call i32 @bar( i32 1 )		; <i32> [#uses=1]
	ret i32 %C.upgrd.1
}

define i32 @foo() {
	%X = call i32 @bar( i32 0 )		; <i32> [#uses=1]
	ret i32 %X
}

