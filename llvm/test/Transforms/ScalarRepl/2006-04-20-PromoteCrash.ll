; RUN: llvm-as < %s | opt -scalarrepl -disable-output

define void @output_toc() {
entry:
	%buf = alloca [256 x i8], align 16		; <[256 x i8]*> [#uses=1]
	%name = alloca i8*, align 4		; <i8**> [#uses=1]
	%real_name = alloca i8*, align 4		; <i8**> [#uses=0]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%buf.upgrd.1 = bitcast [256 x i8]* %buf to i8*		; <i8*> [#uses=1]
	store i8* %buf.upgrd.1, i8** %name
	call void @abort( )
	unreachable
return:		; No predecessors!
	ret void
}

declare void @abort()

