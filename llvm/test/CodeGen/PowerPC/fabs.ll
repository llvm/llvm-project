; RUN: llc < %s -march=ppc32 -mtriple=powerpc-apple-darwin | grep "fabs f1, f1"

define double @fabs(double %f) {
entry:
	%tmp2 = tail call double @fabs( double %f ) readnone	; <double> [#uses=1]
	ret double %tmp2
}
