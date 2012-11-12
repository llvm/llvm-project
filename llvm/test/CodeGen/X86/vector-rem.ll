; RUN: llc < %s -march=x86-64 | grep div | count 8
; RUN: llc < %s -march=x86-64 | grep fmodf | count 4

define <4 x i32> @foo(<4 x i32> %t, <4 x i32> %u) {
	%m = srem <4 x i32> %t, %u
	ret <4 x i32> %m
}
define <4 x i32> @bar(<4 x i32> %t, <4 x i32> %u) {
	%m = urem <4 x i32> %t, %u
	ret <4 x i32> %m
}
define <4 x float> @qux(<4 x float> %t, <4 x float> %u) {
	%m = frem <4 x float> %t, %u
	ret <4 x float> %m
}
