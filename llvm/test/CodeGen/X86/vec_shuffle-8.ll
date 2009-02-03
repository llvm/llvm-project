; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | \
; RUN:   not grep shufps

define void @test(<4 x float>* %res, <4 x float>* %A) {
        %tmp1 = load <4 x float>* %A            ; <<4 x float>> [#uses=1]
        %tmp2 = shufflevector <4 x float> %tmp1, <4 x float> undef, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >          ; <<4 x float>> [#uses=1]
        store <4 x float> %tmp2, <4 x float>* %res
        ret void
}

