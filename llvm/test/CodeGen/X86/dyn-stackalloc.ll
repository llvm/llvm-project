; RUN: llvm-as < %s | llc -march=x86 | not grep 4294967289
; RUN: llvm-as < %s | llc -march=x86 | grep 4294967280
; RUN: llvm-as < %s | llc -march=x86-64 | grep {\\-16}

define void @t() {
A:
	br label %entry

entry:
	%m1 = alloca i32, align 4
	%m2 = alloca [7 x i8], align 16
	call void @s( i32* %m1, [7 x i8]* %m2 )
	ret void
}

declare void @s(i32*, [7 x i8]*)
