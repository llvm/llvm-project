; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -march=x86-64 -relocation-model=static -code-model=small > %t
; RUN: grep leaq %t | count 1

@dst = external global [131072 x i32]
@ptr = external global i32*

define void @off01(i64 %i) nounwind {
entry:
	%.sum = add i64 %i, 16
	%0 = getelementptr [131072 x i32], [131072 x i32]* @dst, i64 0, i64 %.sum
	store i32* %0, i32** @ptr, align 8
	ret void
}
