; RUN: llvm-as < %s | llc -relocation-model=static -mcpu=yonah | grep {andpd.*4(%esp), %xmm}

; The double argument is at 4(esp) which is 16-byte aligned, allowing us to
; fold the load into the andpd.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"
@G = external global double

define void @test({ double, double }* byval  %z, double* %P) {
entry:
	%tmp = getelementptr { double, double }* %z, i32 0, i32 0		; <double*> [#uses=1]
	%tmp1 = load double* %tmp, align 8		; <double> [#uses=1]
	%tmp2 = tail call double @fabs( double %tmp1 )		; <double> [#uses=1]
	%tmp3 = load double* @G, align 16		; <double> [#uses=1]
	%tmp4 = tail call double @fabs( double %tmp3 )		; <double> [#uses=1]
	%tmp6 = add double %tmp4, %tmp2		; <double> [#uses=1]
	store double %tmp6, double* %P, align 8
	ret void
}

declare double @fabs(double)
