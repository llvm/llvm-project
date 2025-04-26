; RUN: llc -mtriple=hexagon -mcpu=hexagonv5 < %s | FileCheck %s
; CHECK: vmpybu
; CHECK: vtrunehb

define <4 x i8> @t_i4x8(<4 x i8> %a, <4 x i8> %b) nounwind {
entry:
	%0 = mul <4 x i8> %a, %b
	ret <4 x i8> %0
}
