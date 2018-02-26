; RUN: llc < %s -mtriple=i686-- -mcpu=yonah

; Legalization example that requires splitting a large vector into smaller pieces.

define void @update(<8 x i32> %val, <8 x i32>* %dst) nounwind {
entry:
	%shl = shl <8 x i32> %val, < i32 2, i32 2, i32 2, i32 2, i32 4, i32 4, i32 4, i32 4 >
	%shr = ashr <8 x i32> %val, < i32 2, i32 2, i32 2, i32 2, i32 4, i32 4, i32 4, i32 4 >
	store <8 x i32> %shr, <8 x i32>* %dst
	ret void
}
