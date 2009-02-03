; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -o %t  -f
; RUN: grep movhlps %t | count 1
; RUN: grep shufps  %t | count 1

define void @test() nounwind {
        %tmp1 = load <4 x float>* null          ; <<4 x float>> [#uses=2]
        %tmp2 = shufflevector <4 x float> %tmp1, <4 x float> < float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00 >, <4 x i32> < i32 0, i32 1, i32 6, i32 7 >             ; <<4 x float>> [#uses=1]
        %tmp3 = shufflevector <4 x float> %tmp1, <4 x float> zeroinitializer, <4 x i32> < i32 2, i32 3, i32 6, i32 7 >                ; <<4 x float>> [#uses=1]
        %tmp4 = add <4 x float> %tmp2, %tmp3            ; <<4 x float>> [#uses=1]
        store <4 x float> %tmp4, <4 x float>* null
        ret void
}

