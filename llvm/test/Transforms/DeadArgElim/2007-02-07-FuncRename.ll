; RUN: llvm-as < %s | opt -deadargelim | llvm-dis | grep {@test(}
; RUN: llvm-as < %s | opt -deadargelim | llvm-dis | not grep dead

define internal i32 @test(i32 %X, i32 %dead) {
	ret i32 %X
}

define i32 @caller() {
	%A = call i32 @test(i32 123, i32 456)
	ret i32 %A
}
