; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: vsubub

define <4 x i8> @t_i4x8(<4 x i8> %a, <4 x i8> %b) nounwind {
entry:
	%0 = sub <4 x i8> %a, %b
	ret <4 x i8> %0
}
