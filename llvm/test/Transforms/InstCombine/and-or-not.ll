; RUN: opt < %s -instcombine -S | grep xor | count 4
; RUN: opt < %s -instcombine -S | not grep and
; RUN: opt < %s -instcombine -S | not grep " or"

; PR1510

; These are all equivalent to A^B

define i32 @test1(i32 %a, i32 %b) {
entry:
        %tmp3 = or i32 %b, %a           ; <i32> [#uses=1]
        %tmp3not = xor i32 %tmp3, -1            ; <i32> [#uses=1]
        %tmp6 = and i32 %b, %a          ; <i32> [#uses=1]
        %tmp7 = or i32 %tmp6, %tmp3not          ; <i32> [#uses=1]
        %tmp7not = xor i32 %tmp7, -1            ; <i32> [#uses=1]
        ret i32 %tmp7not
}

define i32 @test2(i32 %a, i32 %b) {
entry:
        %tmp3 = or i32 %b, %a           ; <i32> [#uses=1]
        %tmp6 = and i32 %b, %a          ; <i32> [#uses=1]
        %tmp6not = xor i32 %tmp6, -1            ; <i32> [#uses=1]
        %tmp7 = and i32 %tmp3, %tmp6not         ; <i32> [#uses=1]
        ret i32 %tmp7
}

define <4 x i32> @test3(<4 x i32> %a, <4 x i32> %b) {
entry:
        %tmp3 = or <4 x i32> %a, %b             ; <<4 x i32>> [#uses=1]
        %tmp3not = xor <4 x i32> %tmp3, < i32 -1, i32 -1, i32 -1, i32 -1 >              ; <<4 x i32>> [#uses=1]
        %tmp6 = and <4 x i32> %a, %b            ; <<4 x i32>> [#uses=1]
        %tmp7 = or <4 x i32> %tmp6, %tmp3not            ; <<4 x i32>> [#uses=1]
        %tmp7not = xor <4 x i32> %tmp7, < i32 -1, i32 -1, i32 -1, i32 -1 >              ; <<4 x i32>> [#uses=1]
        ret <4 x i32> %tmp7not
}

define <4 x i32> @test4(<4 x i32> %a, <4 x i32> %b) {
entry:
        %tmp3 = or <4 x i32> %a, %b             ; <<4 x i32>> [#uses=1]
        %tmp6 = and <4 x i32> %a, %b            ; <<4 x i32>> [#uses=1]
        %tmp6not = xor <4 x i32> %tmp6, < i32 -1, i32 -1, i32 -1, i32 -1 >              ; <<4 x i32>> [#uses=1]
        %tmp7 = and <4 x i32> %tmp3, %tmp6not           ; <<4 x i32>> [#uses=1]
        ret <4 x i32> %tmp7
}

