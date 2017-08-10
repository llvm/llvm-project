; RUN: llc < %s -mtriple=i686-- -mattr=-sse | FileCheck %s -check-prefix=WITHNANS
; RUN: llc < %s -mtriple=i686-- -mattr=-sse -enable-unsafe-fp-math -enable-no-nans-fp-math | FileCheck %s -check-prefix=NONANS

; WITHNANS-LABEL: test:
; WITHNANS: setnp
; NONANS-LABEL: test:
; NONANS-NOT: setnp
define i32 @test(float %f) {
	%tmp = fcmp oeq float %f, 0.000000e+00		; <i1> [#uses=1]
	%tmp.upgrd.1 = zext i1 %tmp to i32		; <i32> [#uses=1]
	ret i32 %tmp.upgrd.1
}

