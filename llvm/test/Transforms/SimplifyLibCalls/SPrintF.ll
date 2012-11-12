; Test that the SPrintFOptimizer works correctly
; RUN: opt < %s -simplify-libcalls -S | \
; RUN:   not grep "call.*sprintf"

; This transformation requires the pointer size, as it assumes that size_t is
; the size of a pointer.
target datalayout = "-p:64:64:64"

@hello = constant [6 x i8] c"hello\00"		; <[6 x i8]*> [#uses=1]
@null = constant [1 x i8] zeroinitializer		; <[1 x i8]*> [#uses=1]
@null_hello = constant [7 x i8] c"\00hello\00"		; <[7 x i8]*> [#uses=1]
@fmt1 = constant [3 x i8] c"%s\00"		; <[3 x i8]*> [#uses=1]
@fmt2 = constant [3 x i8] c"%c\00"		; <[3 x i8]*> [#uses=1]

declare i32 @sprintf(i8*, i8*, ...)

declare i32 @puts(i8*)

define i32 @foo(i8* %p) {
	%target = alloca [1024 x i8]		; <[1024 x i8]*> [#uses=1]
	%target_p = getelementptr [1024 x i8]* %target, i32 0, i32 0		; <i8*> [#uses=7]
	%hello_p = getelementptr [6 x i8]* @hello, i32 0, i32 0		; <i8*> [#uses=2]
	%null_p = getelementptr [1 x i8]* @null, i32 0, i32 0		; <i8*> [#uses=1]
	%nh_p = getelementptr [7 x i8]* @null_hello, i32 0, i32 0		; <i8*> [#uses=1]
	%fmt1_p = getelementptr [3 x i8]* @fmt1, i32 0, i32 0		; <i8*> [#uses=2]
	%fmt2_p = getelementptr [3 x i8]* @fmt2, i32 0, i32 0		; <i8*> [#uses=1]
	store i8 0, i8* %target_p
	%r1 = call i32 (i8*, i8*, ...)* @sprintf( i8* %target_p, i8* %hello_p )		; <i32> [#uses=1]
	%r2 = call i32 (i8*, i8*, ...)* @sprintf( i8* %target_p, i8* %null_p )		; <i32> [#uses=1]
	%r3 = call i32 (i8*, i8*, ...)* @sprintf( i8* %target_p, i8* %nh_p )		; <i32> [#uses=1]
	%r4 = call i32 (i8*, i8*, ...)* @sprintf( i8* %target_p, i8* %fmt1_p, i8* %hello_p )		; <i32> [#uses=1]
	%r4.1 = call i32 (i8*, i8*, ...)* @sprintf( i8* %target_p, i8* %fmt1_p, i8* %p )		; <i32> [#uses=1]
	%r5 = call i32 (i8*, i8*, ...)* @sprintf( i8* %target_p, i8* %fmt2_p, i32 82 )		; <i32> [#uses=1]
	%r6 = add i32 %r1, %r2		; <i32> [#uses=1]
	%r7 = add i32 %r3, %r6		; <i32> [#uses=1]
	%r8 = add i32 %r5, %r7		; <i32> [#uses=1]
	%r9 = add i32 %r8, %r4		; <i32> [#uses=1]
	%r10 = add i32 %r9, %r4.1		; <i32> [#uses=1]
	ret i32 %r10
}
