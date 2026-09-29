; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 < %s
target triple = "powerpc64-unknown-linux-gnu"

define <4 x i16> @test(<4 x float> %f, <4 x float> %g) {
entry:
	%r = fcmp ord <4 x float> %f, %g
	%s = sext <4 x i1> %r to <4 x i16>
	ret <4 x i16> %s
}

define <4 x i16> @test2(<4 x float> %f, <4 x float> %g) {
entry:
	%r = fcmp one <4 x float> %f, %g
	%s = sext <4 x i1> %r to <4 x i16>
	ret <4 x i16> %s
}

