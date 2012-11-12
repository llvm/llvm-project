; This file contains various testcases that check to see that instcombine
; is narrowing computations when possible.
; RUN: opt < %s -instcombine -S | \
; RUN:    grep "ret i1 false"

; test1 - Eliminating the casts in this testcase (by narrowing the AND
; operation) allows instcombine to realize the function always returns false.
;
define i1 @test1(i32 %A, i32 %B) {
        %C1 = icmp slt i32 %A, %B               ; <i1> [#uses=1]
        %ELIM1 = zext i1 %C1 to i32             ; <i32> [#uses=1]
        %C2 = icmp sgt i32 %A, %B               ; <i1> [#uses=1]
        %ELIM2 = zext i1 %C2 to i32             ; <i32> [#uses=1]
        %C3 = and i32 %ELIM1, %ELIM2            ; <i32> [#uses=1]
        %ELIM3 = trunc i32 %C3 to i1            ; <i1> [#uses=1]
        ret i1 %ELIM3
}

