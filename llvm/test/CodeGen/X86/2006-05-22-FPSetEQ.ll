; RUN: llc < %s -mtriple=i686-- -mattr=-sse | FileCheck %s

define i32 @test(float %f) {
; CHECK-LABEL: test:
; CHECK: setnp
	%tmp = fcmp oeq float %f, 0.000000e+00		; <i1> [#uses=1]
	%tmp.upgrd.1 = zext i1 %tmp to i32		; <i32> [#uses=1]
	ret i32 %tmp.upgrd.1
}

define i32 @test_nnan(float %f) {
; CHECK-LABEL: test_nnan:
; CHECK-NOT: setnp
	%tmp = fcmp nnan oeq float %f, 0.000000e+00		; <i1> [#uses=1]
	%tmp.upgrd.1 = zext i1 %tmp to i32		; <i32> [#uses=1]
	ret i32 %tmp.upgrd.1
}
