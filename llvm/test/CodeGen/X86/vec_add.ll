; RUN: llc < %s -march=x86 -mattr=+sse2

define <2 x i64> @test(<2 x i64> %a, <2 x i64> %b) {
entry:
	%tmp9 = add <2 x i64> %b, %a		; <<2 x i64>> [#uses=1]
	ret <2 x i64> %tmp9
}
