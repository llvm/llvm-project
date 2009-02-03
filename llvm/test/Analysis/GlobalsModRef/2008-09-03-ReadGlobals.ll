; RUN: llvm-as < %s | opt -globalsmodref-aa -gvn | llvm-dis | grep call | count 2

@g = internal global i32 0		; <i32*> [#uses=2]

define i32 @r() {
	%tmp = load i32* @g		; <i32> [#uses=1]
	ret i32 %tmp
}

define i32 @f() {
entry:
	%tmp = call i32 @e( )		; <i32> [#uses=1]
	store i32 %tmp, i32* @g
	%tmp2 = call i32 @e( )		; <i32> [#uses=1]
	ret i32 %tmp2
}

declare i32 @e() readonly	; might call @r
