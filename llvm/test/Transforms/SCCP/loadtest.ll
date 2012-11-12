; This test makes sure that these instructions are properly constant propagated.

; RUN: opt < %s -default-data-layout="e-p:32:32" -sccp -S | FileCheck %s
; RUN: opt < %s -default-data-layout="E-p:32:32" -sccp -S | FileCheck %s

; CHECK-NOT: load


@X = constant i32 42		; <i32*> [#uses=1]
@Y = constant [2 x { i32, float }] [ { i32, float } { i32 12, float 1.000000e+00 }, { i32, float } { i32 37, float 0x3FF3B2FEC0000000 } ]		; <[2 x { i32, float }]*> [#uses=2]

define i32 @test1() {
	%B = load i32* @X		; <i32> [#uses=1]
	ret i32 %B
}

define float @test2() {
	%A = getelementptr [2 x { i32, float }]* @Y, i64 0, i64 1, i32 1		; <float*> [#uses=1]
	%B = load float* %A		; <float> [#uses=1]
	ret float %B
}

define i32 @test3() {
	%A = getelementptr [2 x { i32, float }]* @Y, i64 0, i64 0, i32 0		; <i32*> [#uses=1]
	%B = load i32* %A
	ret i32 %B
}

define i8 @test4() {
	%A = bitcast i32* @X to i8*
	%B = load i8* %A
	ret i8 %B
}

