; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep vector_shuffle
; END.

%T = type <4 x float>


define %T @test1(%T %v1) {
  %v2 = shufflevector %T %v1, %T undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret %T %v2
}

define %T @test2(%T %v1) {
  %v2 = shufflevector %T %v1, %T %v1, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  ret %T %v2
}

define float @test3(%T %A, %T %B, float %f) {
        %C = insertelement %T %A, float %f, i32 0
        %D = shufflevector %T %C, %T %B, <4 x i32> <i32 5, i32 0, i32 2, i32 7>
        %E = extractelement %T %D, i32 1
        ret float %E
}

define i32 @test4(<4 x i32> %X) {
        %tmp152.i53899.i = shufflevector <4 x i32> %X, <4 x i32> undef, <4 x i32> zeroinitializer
        %tmp34 = extractelement <4 x i32> %tmp152.i53899.i, i32 0
        ret i32 %tmp34
}

define i32 @test5(<4 x i32> %X) {
        %tmp152.i53899.i = shufflevector <4 x i32> %X, <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 undef, i32 undef>
        %tmp34 = extractelement <4 x i32> %tmp152.i53899.i, i32 0
        ret i32 %tmp34
}

define float @test6(<4 x float> %X) {
        %X1 = bitcast <4 x float> %X to <4 x i32>
        %tmp152.i53899.i = shufflevector <4 x i32> %X1, <4 x i32> undef, <4 x i32> zeroinitializer
        %tmp152.i53900.i = bitcast <4 x i32> %tmp152.i53899.i to <4 x float>
        %tmp34 = extractelement <4 x float> %tmp152.i53900.i, i32 0
        ret float %tmp34
}

define <4 x float> @test7(<4 x float> %tmp45.i) {
        %tmp1642.i = shufflevector <4 x float> %tmp45.i, <4 x float> undef, <4 x i32> < i32 0, i32 1, i32 6, i32 7 >
        ret <4 x float> %tmp1642.i
}
