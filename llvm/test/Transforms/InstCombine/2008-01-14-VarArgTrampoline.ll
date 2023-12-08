; RUN: opt < %s -passes=instcombine -S | grep zeroext

	%struct.FRAME.nest = type { i32, ptr }
	%struct.__builtin_trampoline = type { [10 x i8] }

declare void @llvm.init.trampoline(ptr, ptr, ptr) nounwind 
declare ptr @llvm.adjust.trampoline(ptr) nounwind

declare i32 @f(ptr nest , ...)

define i32 @nest(i32 %n) {
entry:
	%FRAME.0 = alloca %struct.FRAME.nest, align 8		; <ptr> [#uses=3]
	%TRAMP.216 = alloca [10 x i8], align 16		; <ptr> [#uses=1]
	%TRAMP.216.sub = getelementptr [10 x i8], ptr %TRAMP.216, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp3 = getelementptr %struct.FRAME.nest, ptr %FRAME.0, i32 0, i32 0		; <ptr> [#uses=1]
	store i32 %n, ptr %tmp3, align 8
	call void @llvm.init.trampoline( ptr %TRAMP.216.sub, ptr @f, ptr %FRAME.0 )		; <ptr> [#uses=1]
        %tramp = call ptr @llvm.adjust.trampoline( ptr %TRAMP.216.sub)
	%tmp7 = getelementptr %struct.FRAME.nest, ptr %FRAME.0, i32 0, i32 1		; <ptr> [#uses=1]
	store ptr %tramp, ptr %tmp7, align 8
	%tmp2.i = call i32 (...) %tramp( i32 zeroext 0 )		; <i32> [#uses=1]
	ret i32 %tmp2.i
}
