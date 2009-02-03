; RUN: llvm-as < %s | llc -march=x86-64 | grep movd | count 1
; RUN: llvm-as < %s | llc -march=x86-64 | grep {movlhps.*%xmm0, %xmm0}

define <2 x i64> @test3(i64 %A) nounwind {
entry:
	%B = insertelement <2 x i64> undef, i64 %A, i32 1
	ret <2 x i64> %B
}

