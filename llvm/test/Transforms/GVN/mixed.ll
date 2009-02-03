; RUN: llvm-as < %s | opt -gvn | llvm-dis | not grep DEADLOAD
; RUN: llvm-as < %s | opt -gvn | llvm-dis | not grep DEADGEP

define i32 @main(i32** %p) {
block1:
	%z1 = load i32** %p
	%z2 = getelementptr i32* %z1, i32 0
	%z3 = load i32* %z2
	%DEADLOAD = load i32** %p
	%DEADGEP = getelementptr i32* %DEADLOAD, i32 0
	%DEADLOAD2 = load i32* %DEADGEP
	ret i32 %DEADLOAD2
}
