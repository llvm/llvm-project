; RUN: llvm-as < %s | opt -constprop -instcombine | llvm-dis | not grep {call.*llvm.memcpy.i32}

@h = constant [2 x i8] c"h\00"		; <[2 x i8]*> [#uses=1]
@hel = constant [4 x i8] c"hel\00"		; <[4 x i8]*> [#uses=1]
@hello_u = constant [8 x i8] c"hello_u\00"		; <[8 x i8]*> [#uses=1]

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)

define i32 @main() {
	%h_p = getelementptr [2 x i8]* @h, i32 0, i32 0		; <i8*> [#uses=1]
	%hel_p = getelementptr [4 x i8]* @hel, i32 0, i32 0		; <i8*> [#uses=1]
	%hello_u_p = getelementptr [8 x i8]* @hello_u, i32 0, i32 0		; <i8*> [#uses=1]
	%target = alloca [1024 x i8]		; <[1024 x i8]*> [#uses=1]
	%target_p = getelementptr [1024 x i8]* %target, i32 0, i32 0		; <i8*> [#uses=3]
	call void @llvm.memcpy.i32( i8* %target_p, i8* %h_p, i32 2, i32 2 )
	call void @llvm.memcpy.i32( i8* %target_p, i8* %hel_p, i32 4, i32 4 )
	call void @llvm.memcpy.i32( i8* %target_p, i8* %hello_u_p, i32 8, i32 8 )
	ret i32 0
}

