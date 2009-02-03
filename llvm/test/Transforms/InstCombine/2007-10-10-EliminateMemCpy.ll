; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep call
; RUN: llvm-as < %s | opt -std-compile-opts | llvm-dis | not grep xyz

@.str = internal constant [4 x i8] c"xyz\00"		; <[4 x i8]*> [#uses=1]

define void @foo(i8* %P) {
entry:
	%P_addr = alloca i8*		; <i8**> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i8* %P, i8** %P_addr
	%tmp = load i8** %P_addr, align 4		; <i8*> [#uses=1]
	%tmp1 = getelementptr [4 x i8]* @.str, i32 0, i32 0		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32( i8* %tmp, i8* %tmp1, i32 4, i32 1 )
	br label %return

return:		; preds = %entry
	ret void
}

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)
