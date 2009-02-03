; RUN: llvm-as < %s | opt -simplify-libcalls -disable-output

@G = constant [3 x i8] c"%s\00"		; <[3 x i8]*> [#uses=1]

declare i32 @sprintf(i8*, i8*, ...)

define void @foo(i8* %P, i32* %X) {
	call i32 (i8*, i8*, ...)* @sprintf( i8* %P, i8* getelementptr ([3 x i8]* @G, i32 0, i32 0), i32* %X )		; <i32>:1 [#uses=0]
	ret void
}

