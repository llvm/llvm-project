; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -o %t -f
; RUN: grep movlhps %t | count 1
; RUN: grep movhlps %t | count 1

define <4 x float> @test1(<4 x float>* %x, <4 x float>* %y) {
        %tmp = load <4 x float>* %y             ; <<4 x float>> [#uses=2]
        %tmp5 = load <4 x float>* %x            ; <<4 x float>> [#uses=2]
        %tmp9 = add <4 x float> %tmp5, %tmp             ; <<4 x float>> [#uses=1]
        %tmp21 = sub <4 x float> %tmp5, %tmp            ; <<4 x float>> [#uses=1]
        %tmp27 = shufflevector <4 x float> %tmp9, <4 x float> %tmp21, <4 x i32> < i32 0, i32 1, i32 4, i32 5 >                ; <<4 x float>> [#uses=1]
        ret <4 x float> %tmp27
}

define <4 x float> @movhl(<4 x float>* %x, <4 x float>* %y) {
entry:
        %tmp = load <4 x float>* %y             ; <<4 x float>> [#uses=1]
        %tmp3 = load <4 x float>* %x            ; <<4 x float>> [#uses=1]
        %tmp4 = shufflevector <4 x float> %tmp3, <4 x float> %tmp, <4 x i32> < i32 2, i32 3, i32 6, i32 7 >           ; <<4 x float>> [#uses=1]
        ret <4 x float> %tmp4
}

