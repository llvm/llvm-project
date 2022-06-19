; This test makes sure that these instructions are properly constant propagated.

; RUN: opt < %s -passes=ipsccp -S | not grep load
; RUN: opt < %s -passes=ipsccp -S | not grep fdiv

@X = constant i212 42
@Y = constant [2 x { i212, float }] [ { i212, float } { i212 12, float 1.0 }, 
                                     { i212, float } { i212 37, float 0x3FF3B2FEC0000000 } ]
define i212 @test1() {
	%B = load i212, i212* @X
	ret i212 %B
}

define internal float @test2() {
	%A = getelementptr [2 x { i212, float}], [2 x { i212, float}]* @Y, i32 0, i32 1, i32 1
	%B = load float, float* %A
	ret float %B
}

define internal i212 @test3() {
	%A = getelementptr [2 x { i212, float}], [2 x { i212, float}]* @Y, i32 0, i32 0, i32 0
	%B = load i212, i212* %A
	ret i212 %B
}

define float @All()
{
   %A = call float @test2()
   %B = call i212 @test3()
   %C = mul i212 %B, -1234567
   %D = sitofp i212 %C to float
   %E = fdiv float %A, %D
   ret float %E
}


