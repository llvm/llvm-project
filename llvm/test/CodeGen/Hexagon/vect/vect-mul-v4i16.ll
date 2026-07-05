; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK: vmpyh
; CHECK: vmpyh
; CHECK: vtrunewh

define <4 x i16> @t_i4x16(<4 x i16> %a, <4 x i16> %b) nounwind {
entry:
	%0 = mul <4 x i16> %a, %b
	ret <4 x i16> %0
}
