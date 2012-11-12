; RUN: llvm-as %s -o /dev/null

; Dominance relationships is not calculated correctly for unreachable blocks,
; which causes the verifier to barf on this input.

define i32 @test(i1 %b) {
BB0:
        ret i32 7 ; Loop is unreachable

Loop:           ; preds = %L2, %Loop
        %B = phi i32 [ %B, %L2 ], [ %B, %Loop ]         ;PHI has same value always. 
        br i1 %b, label %L2, label %Loop

L2:             ; preds = %Loop
        br label %Loop
}

