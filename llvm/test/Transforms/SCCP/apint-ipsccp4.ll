; This test makes sure that these instructions are properly constant propagated.

; RUN: opt < %s -passes=ipsccp -S | not grep load
; RUN: opt < %s -passes=ipsccp -S | not grep add
; RUN: opt < %s -passes=ipsccp -S | not grep phi


@Y = constant [2 x { i212, float }] [ { i212, float } { i212 12, float 1.0 }, 
                                     { i212, float } { i212 37, float 2.0 } ]

define internal float @test2() {
	%A = getelementptr [2 x { i212, float}], [2 x { i212, float}]* @Y, i32 0, i32 1, i32 1
	%B = load float, float* %A
	ret float %B
}

define internal float  @test3() {
	%A = getelementptr [2 x { i212, float}], [2 x { i212, float}]* @Y, i32 0, i32 0, i32 1
	%B = load float, float* %A
	ret float %B
}

define internal float @test()
{
   %A = call float @test2()
   %B = call float @test3()

   %E = fdiv float %B, %A
   ret float %E
}

define float @All()
{
  %A = call float @test()
  %B = fcmp oge float %A, 1.0
  br i1 %B, label %T, label %F
T:
  %C = fadd float %A, 1.0
  br label %exit
F:
  %D = fadd float %A, 2.0
  br label %exit
exit:
  %E = phi float [%C, %T], [%D, %F]
  ret float %E
}



