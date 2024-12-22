; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: vmpyh
; CHECK: vtrunewh

define <2 x i16> @t_i2x16(<2 x i16> %a, <2 x i16> %b) nounwind {
entry:
	%0 = mul <2 x i16> %a, %b
	ret <2 x i16> %0
}
