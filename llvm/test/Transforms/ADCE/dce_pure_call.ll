; RUN: llvm-as < %s | opt -adce | llvm-dis | not grep call

declare i32 @strlen(i8*) readonly

define void @test() {
	call i32 @strlen( i8* null ) readonly		; <i32>:1 [#uses=0]
	ret void
}
