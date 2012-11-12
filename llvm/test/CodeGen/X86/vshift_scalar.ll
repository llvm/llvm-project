; RUN: llc < %s

; Legalization test that requires scalarizing a vector.

define void @update(<1 x i32> %val, <1 x i32>* %dst) nounwind {
entry:
	%shl = shl <1 x i32> %val, < i32 2>
	%shr = ashr <1 x i32> %val, < i32 4>
	store <1 x i32> %shr, <1 x i32>* %dst
	ret void
}
