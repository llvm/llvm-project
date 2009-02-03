; Test that the StrCmpOptimizer works correctly
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | \
; RUN:   not grep {call.*strcmp}

@hello = constant [6 x i8] c"hello\00"		; <[6 x i8]*> [#uses=1]
@hell = constant [5 x i8] c"hell\00"		; <[5 x i8]*> [#uses=1]
@null = constant [1 x i8] zeroinitializer		; <[1 x i8]*> [#uses=1]

declare i32 @strcmp(i8*, i8*)

declare i32 @puts(i8*)

define i32 @main() {
	%hello_p = getelementptr [6 x i8]* @hello, i32 0, i32 0		; <i8*> [#uses=5]
	%hell_p = getelementptr [5 x i8]* @hell, i32 0, i32 0		; <i8*> [#uses=1]
	%null_p = getelementptr [1 x i8]* @null, i32 0, i32 0		; <i8*> [#uses=4]
	%temp1 = call i32 @strcmp( i8* %hello_p, i8* %hello_p )		; <i32> [#uses=1]
	%temp2 = call i32 @strcmp( i8* %null_p, i8* %null_p )		; <i32> [#uses=1]
	%temp3 = call i32 @strcmp( i8* %hello_p, i8* %null_p )		; <i32> [#uses=1]
	%temp4 = call i32 @strcmp( i8* %null_p, i8* %hello_p )		; <i32> [#uses=1]
	%temp5 = call i32 @strcmp( i8* %hell_p, i8* %hello_p )		; <i32> [#uses=1]
	%rslt1 = add i32 %temp1, %temp2		; <i32> [#uses=1]
	%rslt2 = add i32 %rslt1, %temp3		; <i32> [#uses=1]
	%rslt3 = add i32 %rslt2, %temp4		; <i32> [#uses=1]
	%rslt4 = add i32 %rslt3, %temp5		; <i32> [#uses=1]
	ret i32 %rslt4
}

