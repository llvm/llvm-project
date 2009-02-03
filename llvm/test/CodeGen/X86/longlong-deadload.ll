; RUN: llvm-as < %s | llc -march=x86 | not grep '4{(%...)}
; This should not load or store the top part of *P.

define void @test(i64* %P) nounwind  {
entry:
	%tmp1 = load i64* %P, align 8		; <i64> [#uses=1]
	%tmp2 = xor i64 %tmp1, 1		; <i64> [#uses=1]
	store i64 %tmp2, i64* %P, align 8
	ret void
}

