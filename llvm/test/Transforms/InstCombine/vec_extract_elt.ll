; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep extractelement

define i32 @test(float %f) {
        %tmp7 = insertelement <4 x float> undef, float %f, i32 0                ; <<4 x float>> [#uses=1]
        %tmp17 = bitcast <4 x float> %tmp7 to <4 x i32>         ; <<4 x i32>> [#uses=1]
        %tmp19 = extractelement <4 x i32> %tmp17, i32 0         ; <i32> [#uses=1]
        ret i32 %tmp19
}

