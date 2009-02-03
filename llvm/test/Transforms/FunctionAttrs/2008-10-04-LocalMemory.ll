; RUN: llvm-as < %s | opt -functionattrs | llvm-dis | grep readnone | count 2

declare i32 @g(i32*) readnone

define i32 @f() {
	%x = alloca i32		; <i32*> [#uses=2]
	store i32 0, i32* %x
	%y = call i32 @g(i32* %x)		; <i32> [#uses=1]
	ret i32 %y
}
