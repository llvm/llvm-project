; RUN: llvm-as < %s | opt -ipsccp | llvm-dis | \
; RUN:   grep -v {ret i32 17} | grep -v {ret i32 undef} | not grep ret

define internal i32 @bar(i32 %A) {
	%X = add i32 1, 2		; <i32> [#uses=0]
	ret i32 %A
}

define i32 @foo() {
	%X = call i32 @bar( i32 17 )		; <i32> [#uses=1]
	ret i32 %X
}

