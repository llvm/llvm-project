; Test that the StrCatOptimizer works correctly
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | \
; RUN:    not grep {call.*strlen}

target datalayout = "e-p:32:32"
@hello = constant [6 x i8] c"hello\00"		; <[6 x i8]*> [#uses=3]
@null = constant [1 x i8] zeroinitializer		; <[1 x i8]*> [#uses=3]
@null_hello = constant [7 x i8] c"\00hello\00"		; <[7 x i8]*> [#uses=1]

declare i32 @strlen(i8*)

define i32 @test1() {
	%hello_p = getelementptr [6 x i8]* @hello, i32 0, i32 0		; <i8*> [#uses=1]
	%hello_l = call i32 @strlen( i8* %hello_p )		; <i32> [#uses=1]
	ret i32 %hello_l
}

define i32 @test2() {
	%null_p = getelementptr [1 x i8]* @null, i32 0, i32 0		; <i8*> [#uses=1]
	%null_l = call i32 @strlen( i8* %null_p )		; <i32> [#uses=1]
	ret i32 %null_l
}

define i32 @test3() {
	%null_hello_p = getelementptr [7 x i8]* @null_hello, i32 0, i32 0		; <i8*> [#uses=1]
	%null_hello_l = call i32 @strlen( i8* %null_hello_p )		; <i32> [#uses=1]
	ret i32 %null_hello_l
}

define i1 @test4() {
	%hello_p = getelementptr [6 x i8]* @hello, i32 0, i32 0		; <i8*> [#uses=1]
	%hello_l = call i32 @strlen( i8* %hello_p )		; <i32> [#uses=1]
	%eq_hello = icmp eq i32 %hello_l, 0		; <i1> [#uses=1]
	ret i1 %eq_hello
}

define i1 @test5() {
	%null_p = getelementptr [1 x i8]* @null, i32 0, i32 0		; <i8*> [#uses=1]
	%null_l = call i32 @strlen( i8* %null_p )		; <i32> [#uses=1]
	%eq_null = icmp eq i32 %null_l, 0		; <i1> [#uses=1]
	ret i1 %eq_null
}

define i1 @test6() {
	%hello_p = getelementptr [6 x i8]* @hello, i32 0, i32 0		; <i8*> [#uses=1]
	%hello_l = call i32 @strlen( i8* %hello_p )		; <i32> [#uses=1]
	%ne_hello = icmp ne i32 %hello_l, 0		; <i1> [#uses=1]
	ret i1 %ne_hello
}

define i1 @test7() {
	%null_p = getelementptr [1 x i8]* @null, i32 0, i32 0		; <i8*> [#uses=1]
	%null_l = call i32 @strlen( i8* %null_p )		; <i32> [#uses=1]
	%ne_null = icmp ne i32 %null_l, 0		; <i1> [#uses=1]
	ret i1 %ne_null
}
