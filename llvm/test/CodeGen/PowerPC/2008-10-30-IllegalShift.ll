; RUN: llvm-as < %s | llc -march=ppc32
; PR2986
@argc = external global i32		; <i32*> [#uses=1]
@buffer = external global [32 x i8], align 4		; <[32 x i8]*> [#uses=1]

define void @test1() nounwind noinline {
entry:
	%0 = load i32* @argc, align 4		; <i32> [#uses=1]
	%1 = trunc i32 %0 to i8		; <i8> [#uses=1]
	tail call void @llvm.memset.i32(i8* getelementptr ([32 x i8]* @buffer, i32 0, i32 0), i8 %1, i32 17, i32 4)
	unreachable
}

declare void @llvm.memset.i32(i8*, i8, i32, i32) nounwind
