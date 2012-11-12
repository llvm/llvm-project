; RUN: llc < %s -mcpu=yonah | grep cvtss2sd
; RUN: llc < %s -mcpu=yonah | grep fstps
; RUN: llc < %s -mcpu=yonah | not grep cvtsd2ss

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-apple-darwin8"

define void @test(double *%b) {
entry:
	%tmp13 = tail call double @foo()
	%tmp1314 = fptrunc double %tmp13 to float		; <float> [#uses=1]
	%tmp3940 = fpext float %tmp1314 to double		; <double> [#uses=1]
	store volatile double %tmp3940, double* %b
	ret void
}

declare double @foo()
