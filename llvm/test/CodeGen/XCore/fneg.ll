; RUN: llc < %s -march=xcore | FileCheck %s
define i1 @test(double %F) nounwind {
entry:
; CHECK: test:
; CHECK: xor
	%0 = fsub double -0.000000e+00, %F
	%1 = fcmp olt double 0.000000e+00, %0
	ret i1 %1
}
