; RUN: llc -mtriple=hexagon -mcpu=hexagonv5 < %s | FileCheck %s
; CHECK: vmpybu
; CHECK: vmpybu

define <8 x i8> @t_i8x8(<8 x i8> %a, <8 x i8> %b) nounwind {
entry:
	%0 = mul <8 x i8> %a, %b
	ret <8 x i8> %0
}
