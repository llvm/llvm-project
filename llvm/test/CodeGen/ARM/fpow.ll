; RUN: llvm-as < %s | llc -march=arm
; RUN: llvm-as < %s | llc -march=thumb

define double @t(double %x, double %y) nounwind optsize {
entry:
	%0 = tail call double @llvm.pow.f64( double %x, double %y )		; <double> [#uses=1]
	ret double %0
}

declare double @llvm.pow.f64(double, double) nounwind readonly
