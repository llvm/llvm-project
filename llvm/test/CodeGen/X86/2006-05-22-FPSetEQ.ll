; RUN: llvm-as < %s | llc -march=x86 | grep setnp
; RUN: llvm-as < %s | llc -march=x86 -enable-unsafe-fp-math | \
; RUN:   not grep setnp

define i32 @test(float %f) {
	%tmp = fcmp oeq float %f, 0.000000e+00		; <i1> [#uses=1]
	%tmp.upgrd.1 = zext i1 %tmp to i32		; <i32> [#uses=1]
	ret i32 %tmp.upgrd.1
}

