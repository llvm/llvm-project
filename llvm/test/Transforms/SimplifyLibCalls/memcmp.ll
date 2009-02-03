; Test that the memcmpOptimizer works correctly
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | not grep {call.*memcmp}

@h = constant [2 x i8] c"h\00"		; <[2 x i8]*> [#uses=0]
@hel = constant [4 x i8] c"hel\00"		; <[4 x i8]*> [#uses=0]
@hello_u = constant [8 x i8] c"hello_u\00"		; <[8 x i8]*> [#uses=0]

declare i32 @memcmp(i8*, i8*, i32)

define void @test(i8* %P, i8* %Q, i32 %N, i32* %IP, i1* %BP) {
	%A = call i32 @memcmp( i8* %P, i8* %P, i32 %N )		; <i32> [#uses=1]
	volatile store i32 %A, i32* %IP
	%B = call i32 @memcmp( i8* %P, i8* %Q, i32 0 )		; <i32> [#uses=1]
	volatile store i32 %B, i32* %IP
	%C = call i32 @memcmp( i8* %P, i8* %Q, i32 1 )		; <i32> [#uses=1]
	volatile store i32 %C, i32* %IP
	%D = call i32 @memcmp( i8* %P, i8* %Q, i32 2 )		; <i32> [#uses=1]
	%E = icmp eq i32 %D, 0		; <i1> [#uses=1]
	volatile store i1 %E, i1* %BP
	ret void
}

