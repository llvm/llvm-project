; RUN: llc < %s -mtriple=xcore | FileCheck %s
define i1 @test(double %F, double %G) nounwind {
entry:
; CHECK-LABEL: test:
; CHECK: xor
	%0 = fsub double -0.000000e+00, %F
	%1 = fcmp olt double %G, %0
	ret i1 %1
}
