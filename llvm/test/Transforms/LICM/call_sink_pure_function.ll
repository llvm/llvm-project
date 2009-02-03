; RUN: llvm-as < %s | opt -basicaa -licm | llvm-dis | %prcontext strlen 1 | grep Out:

declare i32 @strlen(i8*) readonly

declare void @foo()

define i32 @test(i8* %P) {
	br label %Loop

Loop:		; preds = %Loop, %0
	%A = call i32 @strlen( i8* %P ) readonly		; <i32> [#uses=1]
	br i1 false, label %Loop, label %Out

Out:		; preds = %Loop
	ret i32 %A
}
