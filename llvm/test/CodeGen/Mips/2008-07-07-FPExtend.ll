; RUN: llc -mtriple=mips -mattr=single-float  < %s | FileCheck %s

define double @dofloat(float %a) nounwind {
entry:
; CHECK: __extendsfdf2
	fpext float %a to double		; <double>:0 [#uses=1]
	ret double %0
}
