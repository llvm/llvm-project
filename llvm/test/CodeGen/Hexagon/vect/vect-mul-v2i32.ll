; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: mpyi
; CHECK: mpyi

define <2 x i32> @t_i2x32(<2 x i32> %a, <2 x i32> %b) nounwind {
entry:
	%0 = mul <2 x i32> %a, %b
	ret <2 x i32> %0
}
