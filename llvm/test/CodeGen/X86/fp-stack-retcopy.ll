; This should not copy the result of foo into an xmm register.
; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah -mtriple=i686-apple-darwin9 | not grep xmm
; rdar://5689903

declare double @foo()

define double @carg({ double, double }* byval  %z) nounwind  {
entry:
	%tmp5 = tail call double @foo() nounwind 		; <double> [#uses=1]
	ret double %tmp5
}

