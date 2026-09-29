; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll


define i39 @test_extractelement(<4 x i39> %V) {
        %R = extractelement <4 x i39> %V, i32 1
        ret i39 %R
}

define <4 x i39> @test_insertelement(<4 x i39> %V) {
        %R = insertelement <4 x i39> %V, i39 0, i32 0
        ret <4 x i39> %R
}

define <4 x i39> @test_shufflevector_u(<4 x i39> %V) {
        %R = shufflevector <4 x i39> %V, <4 x i39> %V, 
                  <4 x i32> < i32 1, i32 undef, i32 7, i32 2>
        ret <4 x i39> %R
}

define <4 x float> @test_shufflevector_f(<4 x float> %V) {
        %R = shufflevector <4 x float> %V, <4 x float> undef, 
                  <4 x i32> < i32 1, i32 undef, i32 7, i32 2>
        ret <4 x float> %R
}
