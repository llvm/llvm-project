; RUN: llvm-as < %s | opt -sccp | llvm-dis | not grep select

define i32 @test1(i1 %C) {
	%X = select i1 %C, i32 0, i32 0		; <i32> [#uses=1]
	ret i32 %X
}

define i32 @test2(i1 %C) {
	%X = select i1 %C, i32 0, i32 undef		; <i32> [#uses=1]
	ret i32 %X
}

