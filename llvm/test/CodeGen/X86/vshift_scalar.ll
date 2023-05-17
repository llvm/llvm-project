; RUN: llc < %s
; REQUIRES: default_triple

; Legalization test that requires scalarizing a vector.

define void @update(<1 x i32> %val, ptr %dst) nounwind {
entry:
	%shl = shl <1 x i32> %val, < i32 2>
	%shr = ashr <1 x i32> %val, < i32 4>
	store <1 x i32> %shr, ptr %dst
	ret void
}
