; This test ensures that the simplifycfg pass continues to constant fold
; terminator instructions.

; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | not grep br

define i32 @test(i32 %A, i32 %B) {
J:
        %C = add i32 %A, 12             ; <i32> [#uses=2]
        br i1 true, label %L, label %K
L:              ; preds = %J
        %D = add i32 %C, %B             ; <i32> [#uses=1]
        ret i32 %D
K:              ; preds = %J
        %E = add i32 %C, %B             ; <i32> [#uses=1]
        ret i32 %E
}

