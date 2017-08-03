; RUN: llc < %s -mtriple=i686-- -mcpu=yonah | not grep pxor

; This should not need to materialize 0.0 to evaluate the condition.

define i32 @test(double %X) nounwind  {
entry:
	%tmp6 = fcmp uno double %X, 0.000000e+00		; <i1> [#uses=1]
	%tmp67 = zext i1 %tmp6 to i32		; <i32> [#uses=1]
	ret i32 %tmp67
}

