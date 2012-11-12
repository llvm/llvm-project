; RUN: opt < %s -instcombine -S | grep zeroext

	%struct.FRAME.nest = type { i32, i32 (...)* }
	%struct.__builtin_trampoline = type { [10 x i8] }

declare void @llvm.init.trampoline(i8*, i8*, i8*) nounwind 
declare i8* @llvm.adjust.trampoline(i8*) nounwind

declare i32 @f(%struct.FRAME.nest* nest , ...)

define i32 @nest(i32 %n) {
entry:
	%FRAME.0 = alloca %struct.FRAME.nest, align 8		; <%struct.FRAME.nest*> [#uses=3]
	%TRAMP.216 = alloca [10 x i8], align 16		; <[10 x i8]*> [#uses=1]
	%TRAMP.216.sub = getelementptr [10 x i8]* %TRAMP.216, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp3 = getelementptr %struct.FRAME.nest* %FRAME.0, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 %n, i32* %tmp3, align 8
	%FRAME.06 = bitcast %struct.FRAME.nest* %FRAME.0 to i8*		; <i8*> [#uses=1]
	call void @llvm.init.trampoline( i8* %TRAMP.216.sub, i8* bitcast (i32 (%struct.FRAME.nest*, ...)* @f to i8*), i8* %FRAME.06 )		; <i8*> [#uses=1]
        %tramp = call i8* @llvm.adjust.trampoline( i8* %TRAMP.216.sub)
	%tmp7 = getelementptr %struct.FRAME.nest* %FRAME.0, i32 0, i32 1		; <i32 (...)**> [#uses=1]
	%tmp89 = bitcast i8* %tramp to i32 (...)*		; <i32 (...)*> [#uses=2]
	store i32 (...)* %tmp89, i32 (...)** %tmp7, align 8
	%tmp2.i = call i32 (...)* %tmp89( i32 zeroext 0 )		; <i32> [#uses=1]
	ret i32 %tmp2.i
}
