; RUN: llvm-as < %s | opt -lowersetjmp | llvm-dis | grep invoke

	%JmpBuf = type i32
@.str_1 = internal constant [13 x i8] c"returned %d\0A\00"		; <[13 x i8]*> [#uses=1]

declare void @llvm.longjmp(i32*, i32)

declare i32 @llvm.setjmp(i32*)

declare void @foo()

define i32 @simpletest() {
	%B = alloca i32		; <i32*> [#uses=2]
	%Val = call i32 @llvm.setjmp( i32* %B )		; <i32> [#uses=2]
	%V = icmp ne i32 %Val, 0		; <i1> [#uses=1]
	br i1 %V, label %LongJumped, label %Normal
Normal:		; preds = %0
	call void @foo( )
	call void @llvm.longjmp( i32* %B, i32 42 )
	ret i32 0
LongJumped:		; preds = %0
	ret i32 %Val
}

declare i32 @printf(i8*, ...)

define i32 @main() {
	%V = call i32 @simpletest( )		; <i32> [#uses=1]
	call i32 (i8*, ...)* @printf( i8* getelementptr ([13 x i8]* @.str_1, i64 0, i64 0), i32 %V )		; <i32>:1 [#uses=0]
	ret i32 0
}

