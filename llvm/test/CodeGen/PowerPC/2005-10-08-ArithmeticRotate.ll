; This was erroneously being turned into an rlwinm instruction.
; The sign bit does matter in this case.

; RUN: llc -verify-machineinstrs < %s -march=ppc32 | grep srawi

define i32 @test(i32 %X) {
        %Y = and i32 %X, -2             ; <i32> [#uses=1]
        %Z = ashr i32 %Y, 11            ; <i32> [#uses=1]
        ret i32 %Z
}

