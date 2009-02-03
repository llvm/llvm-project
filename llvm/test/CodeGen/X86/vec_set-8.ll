; RUN: llvm-as < %s | llc -march=x86-64 | not grep movsd
; RUN: llvm-as < %s | llc -march=x86-64 | grep {movd.*%rdi,.*%xmm0}

define <2 x i64> @test(i64 %i) nounwind  {
entry:
	%tmp10 = insertelement <2 x i64> undef, i64 %i, i32 0
	%tmp11 = insertelement <2 x i64> %tmp10, i64 0, i32 1
	ret <2 x i64> %tmp11
}

