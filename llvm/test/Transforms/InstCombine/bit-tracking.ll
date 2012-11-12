; This file contains various testcases that require tracking whether bits are
; set or cleared by various instructions.
; RUN: opt < %s -instcombine -instcombine -S |\
; RUN:   not grep %ELIM

; Reduce down to a single XOR
define i32 @test3(i32 %B) {
        %ELIMinc = and i32 %B, 1                ; <i32> [#uses=1]
        %tmp.5 = xor i32 %ELIMinc, 1            ; <i32> [#uses=1]
        %ELIM7 = and i32 %B, -2         ; <i32> [#uses=1]
        %tmp.8 = or i32 %tmp.5, %ELIM7          ; <i32> [#uses=1]
        ret i32 %tmp.8
}

; Finally, a bigger case where we chain things together.  This corresponds to
; incrementing a single-bit bitfield, which should become just an xor.
define i32 @test4(i32 %B) {
        %ELIM3 = shl i32 %B, 31         ; <i32> [#uses=1]
        %ELIM4 = ashr i32 %ELIM3, 31            ; <i32> [#uses=1]
        %inc = add i32 %ELIM4, 1                ; <i32> [#uses=1]
        %ELIM5 = and i32 %inc, 1                ; <i32> [#uses=1]
        %ELIM7 = and i32 %B, -2         ; <i32> [#uses=1]
        %tmp.8 = or i32 %ELIM5, %ELIM7          ; <i32> [#uses=1]
        ret i32 %tmp.8
}

