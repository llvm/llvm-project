; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 > %t
; RUN: grep shuf %t | count 2
; RUN: not grep unpck %t

define void @test(<4 x float>* %res, <4 x float>* %A, <4 x float>* %B, <4 x float>* %C) {
        %tmp3 = load <4 x float>* %B            ; <<4 x float>> [#uses=1]
        %tmp5 = load <4 x float>* %C            ; <<4 x float>> [#uses=1]
        %tmp11 = shufflevector <4 x float> %tmp3, <4 x float> %tmp5, <4 x i32> < i32 1, i32 4, i32 1, i32 5 >         ; <<4 x float>> [#uses=1]
        store <4 x float> %tmp11, <4 x float>* %res
        ret void
}

