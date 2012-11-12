; RUN: llc < %s | not grep orpd
; RUN: llc < %s | grep andpd | count 1

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin8"

define double @test(double %X) nounwind  {
entry:
	%tmp2 = tail call double @copysign( double 0.000000e+00, double %X ) nounwind readnone 		; <double> [#uses=1]
	ret double %tmp2
}

declare double @copysign(double, double) nounwind readnone 

