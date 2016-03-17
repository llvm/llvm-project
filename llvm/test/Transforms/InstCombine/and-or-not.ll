; RUN: opt < %s -instcombine -S | FileCheck %s

; PR1510

; These are all equivalent to A^B

define i32 @test1(i32 %a, i32 %b) {
        %tmp3 = or i32 %b, %a           ; <i32> [#uses=1]
        %tmp3not = xor i32 %tmp3, -1            ; <i32> [#uses=1]
        %tmp6 = and i32 %b, %a          ; <i32> [#uses=1]
        %tmp7 = or i32 %tmp6, %tmp3not          ; <i32> [#uses=1]
        %tmp7not = xor i32 %tmp7, -1            ; <i32> [#uses=1]
        ret i32 %tmp7not

; CHECK-LABEL: @test1(
; CHECK-NEXT:    [[TMP7NOT:%.*]] = xor i32 %b, %a
; CHECK-NEXT:    ret i32 [[TMP7NOT]]
}

define i32 @test2(i32 %a, i32 %b) {
        %tmp3 = or i32 %b, %a           ; <i32> [#uses=1]
        %tmp6 = and i32 %b, %a          ; <i32> [#uses=1]
        %tmp6not = xor i32 %tmp6, -1            ; <i32> [#uses=1]
        %tmp7 = and i32 %tmp3, %tmp6not         ; <i32> [#uses=1]
        ret i32 %tmp7

; CHECK-LABEL: @test2(
; CHECK-NEXT:    [[TMP7:%.*]] = xor i32 %b, %a
; CHECK-NEXT:    ret i32 [[TMP7]]
}

define <4 x i32> @test3(<4 x i32> %a, <4 x i32> %b) {
        %tmp3 = or <4 x i32> %a, %b             ; <<4 x i32>> [#uses=1]
        %tmp3not = xor <4 x i32> %tmp3, < i32 -1, i32 -1, i32 -1, i32 -1 >              ; <<4 x i32>> [#uses=1]
        %tmp6 = and <4 x i32> %a, %b            ; <<4 x i32>> [#uses=1]
        %tmp7 = or <4 x i32> %tmp6, %tmp3not            ; <<4 x i32>> [#uses=1]
        %tmp7not = xor <4 x i32> %tmp7, < i32 -1, i32 -1, i32 -1, i32 -1 >              ; <<4 x i32>> [#uses=1]
        ret <4 x i32> %tmp7not

; CHECK-LABEL: @test3(
; CHECK-NEXT:    [[TMP7NOT:%.*]] = xor <4 x i32> %a, %b
; CHECK-NEXT:    ret <4 x i32> [[TMP7NOT]]
}

define <4 x i32> @test4(<4 x i32> %a, <4 x i32> %b) {
        %tmp3 = or <4 x i32> %a, %b             ; <<4 x i32>> [#uses=1]
        %tmp6 = and <4 x i32> %a, %b            ; <<4 x i32>> [#uses=1]
        %tmp6not = xor <4 x i32> %tmp6, < i32 -1, i32 -1, i32 -1, i32 -1 >              ; <<4 x i32>> [#uses=1]
        %tmp7 = and <4 x i32> %tmp3, %tmp6not           ; <<4 x i32>> [#uses=1]
        ret <4 x i32> %tmp7

; CHECK-LABEL: @test4(
; CHECK-NEXT:    [[TMP7:%.*]] = xor <4 x i32> %a, %b
; CHECK-NEXT:    ret <4 x i32> [[TMP7]]
}

