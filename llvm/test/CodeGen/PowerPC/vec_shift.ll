; RUN: llc -verify-machineinstrs < %s  -mtriple=ppc32-- -mcpu=g5
; PR3628

define void @update(<4 x i32> %val, ptr %dst) nounwind {
entry:
	%shl = shl <4 x i32> %val, < i32 4, i32 3, i32 2, i32 1 >
	%shr = ashr <4 x i32> %shl, < i32 1, i32 2, i32 3, i32 4 >
	store <4 x i32> %shr, ptr %dst
	ret void
}
