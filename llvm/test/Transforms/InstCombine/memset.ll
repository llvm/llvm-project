; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep {call.*llvm.memset}

declare void @llvm.memset.i32(i8*, i8, i32, i32)

define i32 @main() {
	%target = alloca [1024 x i8]		; <[1024 x i8]*> [#uses=1]
	%target_p = getelementptr [1024 x i8]* %target, i32 0, i32 0		; <i8*> [#uses=5]
	call void @llvm.memset.i32( i8* %target_p, i8 1, i32 0, i32 1 )
	call void @llvm.memset.i32( i8* %target_p, i8 1, i32 1, i32 1 )
	call void @llvm.memset.i32( i8* %target_p, i8 1, i32 2, i32 2 )
	call void @llvm.memset.i32( i8* %target_p, i8 1, i32 4, i32 4 )
	call void @llvm.memset.i32( i8* %target_p, i8 1, i32 8, i32 8 )
	ret i32 0
}

