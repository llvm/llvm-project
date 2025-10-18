; RUN: llc < %s -mtriple=i686-- -mattr=-sse | FileCheck %s

; CHECK-LABEL: test:
; CHECK: setnp
define i32 @test(float %f) {
	%tmp = fcmp oeq float %f, 0.000000e+00		; <i1> [#uses=1]
	%tmp.upgrd.1 = zext i1 %tmp to i32		; <i32> [#uses=1]
	ret i32 %tmp.upgrd.1
}

; CHECK-LABEL: test_nnan:
; CHECK-NOT: setnp
define i32 @test_nnan(float %f) {
	%tmp = fcmp nnan oeq float %f, 0.000000e+00		; <i1> [#uses=1]
	%tmp.upgrd.1 = zext i1 %tmp to i32		; <i32> [#uses=1]
	ret i32 %tmp.upgrd.1
}
