; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK-NOT: r1:0 = combine(r1, r0)

define <4 x i8> @t_i4x8(<4 x i8> %a, <4 x i8> %b) nounwind {
entry:
	%0 = mul <4 x i8> %a, %b
	ret <4 x i8> %0
}
