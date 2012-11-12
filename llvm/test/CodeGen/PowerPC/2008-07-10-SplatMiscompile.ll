; RUN: llc < %s -march=ppc32 -mcpu=g5 | grep vadduhm
; RUN: llc < %s -march=ppc32 -mcpu=g5 | grep vsubuhm

define <4 x i32> @test() nounwind {
	ret <4 x i32> < i32 4293066722, i32 4293066722, i32 4293066722, i32 4293066722>
}

define <4 x i32> @test2() nounwind {
	ret <4 x i32> < i32 1114129, i32 1114129, i32 1114129, i32 1114129>
}
