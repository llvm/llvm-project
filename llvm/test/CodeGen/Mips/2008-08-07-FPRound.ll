; RUN: llc -march=mips -mattr=single-float  < %s | FileCheck %s

define float @round2float(double %a) nounwind {
entry:
; CHECK: __truncdfsf2
	fptrunc double %a to float		; <float>:0 [#uses=1]
	ret float %0
}
