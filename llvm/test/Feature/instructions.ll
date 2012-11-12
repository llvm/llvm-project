; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

define i32 @test_extractelement(<4 x i32> %V) {
        %R = extractelement <4 x i32> %V, i32 1         ; <i32> [#uses=1]
        ret i32 %R
}

define <4 x i32> @test_insertelement(<4 x i32> %V) {
        %R = insertelement <4 x i32> %V, i32 0, i32 0           ; <<4 x i32>> [#uses=1]
        ret <4 x i32> %R
}

define <4 x i32> @test_shufflevector_u(<4 x i32> %V) {
        %R = shufflevector <4 x i32> %V, <4 x i32> %V, <4 x i32> < i32 1, i32 undef, i32 7, i32 2 >             ; <<4 x i32>> [#uses=1]
        ret <4 x i32> %R
}

define <4 x float> @test_shufflevector_f(<4 x float> %V) {
        %R = shufflevector <4 x float> %V, <4 x float> undef, <4 x i32> < i32 1, i32 undef, i32 7, i32 2 >      ; <<4 x float>> [#uses=1]
        ret <4 x float> %R
}

