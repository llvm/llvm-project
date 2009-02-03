; RUN: llvm-as < %s | llc -march=c
@.str15 = external global [2 x i8]

define <4 x i32> @foo(<4 x i32> %a, i32 %b) {
  %c = insertelement <4 x i32> %a, i32 1, i32 %b
  
  ret <4 x i32> %c
}

define i32 @test2(<4 x i32> %a, i32 %b) {
  %c = extractelement <4 x i32> %a, i32 1
  
  ret i32 %c
}

define <4 x float> @test3(<4 x float> %Y) {
	%Z = add <4 x float> %Y, %Y
	%X = shufflevector <4 x float> zeroinitializer, <4 x float> %Z, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >
	ret <4 x float> %X
}

define void @test4() {
	%x = alloca <4 x float>
	%tmp3.i16 = getelementptr <4 x float>* %x, i32 0, i32 0
	store float 1.0, float* %tmp3.i16
	ret void
}

define i32* @test5({i32, i32} * %P) {
	%x = getelementptr {i32, i32} * %P, i32 0, i32 1
	ret i32* %x
}

define i8* @test6() {
  ret i8* getelementptr ([2 x i8]* @.str15, i32 0, i32 0) 
}

