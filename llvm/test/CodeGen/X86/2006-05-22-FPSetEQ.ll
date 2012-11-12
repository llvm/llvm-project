; RUN: llc < %s -march=x86 -mattr=-sse | grep setnp
; RUN: llc < %s -march=x86 -mattr=-sse -enable-unsafe-fp-math -enable-no-nans-fp-math | \
; RUN:   not grep setnp

define i32 @test(float %f) {
	%tmp = fcmp oeq float %f, 0.000000e+00		; <i1> [#uses=1]
	%tmp.upgrd.1 = zext i1 %tmp to i32		; <i32> [#uses=1]
	ret i32 %tmp.upgrd.1
}

