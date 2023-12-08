; RUN: llc -march=mips -mattr=single-float  < %s | FileCheck %s

define double @dofloat(double %a, double %b) nounwind {
entry:
; CHECK: __adddf3
	fadd double %a, %b		; <double>:0 [#uses=1]
	ret double %0
}
