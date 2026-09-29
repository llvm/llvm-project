; RUN: llc < %s -mtriple=thumb-apple-darwin

%struct.rtx_def = type { i8 }
@str = external global [7 x i8]

define void @f1() {
	%D = alloca %struct.rtx_def, align 1
	%tmp7 = load i32, ptr %D
	%tmp14 = lshr i32 %tmp7, 1
	%tmp1415 = and i32 %tmp14, 1
	call void (i32, ...) @printf( i32 undef, i32 0, i32 %tmp1415 )
	ret void
}

declare void @printf(i32, ...)
