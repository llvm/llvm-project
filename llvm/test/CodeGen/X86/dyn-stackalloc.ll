; RUN: llc < %s -mtriple=i686-linux | FileCheck %s -check-prefix=X32
; X32-NOT:     {{$429496728|-7}}
; X32:     {{$4294967280|-16}}
; X32-NOT:     {{$429496728|-7}}
; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s -check-prefix=X64
; X64:     -16

define void @t() nounwind {
A:
	br label %entry

entry:
	%m1 = alloca i32, align 4
	%m2 = alloca [7 x i8], align 16
	call void @s( i32* %m1, [7 x i8]* %m2 )
	ret void
}

declare void @s(i32*, [7 x i8]*)
