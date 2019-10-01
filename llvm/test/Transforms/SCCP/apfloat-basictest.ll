; This is a basic sanity check for constant propagation. The fneg instruction
; should be eliminated.

; RUN: opt < %s -sccp -S | FileCheck %s

define double @test(i1 %B) {
	br i1 %B, label %BB1, label %BB2
BB1:
	%Val = fneg double 42.0
	br label %BB3
BB2:
	br label %BB3
BB3:
	%Ret = phi double [%Val, %BB1], [1.0, %BB2]
	ret double %Ret
; CHECK-LABEL: @test(
; CHECK: [[PHI:%.*]] = phi double [ -4.200000e+01, %BB1 ], [ 1.000000e+00, %BB2 ]
}

define double @test1(i1 %B) {
        br i1 %B, label %BB1, label %BB2
BB1:
        %Div = fdiv double 1.0, 1.0
        %Val = fneg double %Div
        br label %BB3
BB2:
        br label %BB3
BB3:
        %Ret = phi double [%Val, %BB1], [1.0, %BB2]
        ret double %Ret
; CHECK-LABEL: @test1(
; CHECK: [[PHI:%.*]] = phi double [ -1.000000e+00, %BB1 ], [ 1.000000e+00, %BB2 ]
}
