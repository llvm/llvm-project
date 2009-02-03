; RUN: llvm-as < %s | opt -globalsmodref-aa -gvn | llvm-dis | not grep load

; This test requires the use of previous analyses to determine that
; doesnotmodX does not modify X (because 'sin' doesn't).

@X = internal global i32 4		; <i32*> [#uses=2]

declare double @sin(double) readnone

define i32 @test(i32* %P) {
	store i32 12, i32* @X
	call double @doesnotmodX( double 1.000000e+00 )		; <double>:1 [#uses=0]
	%V = load i32* @X		; <i32> [#uses=1]
	ret i32 %V
}

define double @doesnotmodX(double %V) {
	%V2 = call double @sin( double %V ) readnone		; <double> [#uses=1]
	ret double %V2
}
