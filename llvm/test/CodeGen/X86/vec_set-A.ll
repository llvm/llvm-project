; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep {movl.*\$1, %}
define <2 x i64> @test1() nounwind {
entry:
	ret <2 x i64> < i64 1, i64 0 >
}

