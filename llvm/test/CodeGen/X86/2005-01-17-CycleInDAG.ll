; This testcase was distilled from 132.ijpeg.  Bsaically we cannot fold the
; load into the sub instruction here as it induces a cycle in the dag, which
; is invalid code (there is no correct way to order the instruction).  Check
; that we do not fold the load into the sub.

; RUN: llc < %s -march=x86 | not grep sub.*GLOBAL

@GLOBAL = external global i32           ; <i32*> [#uses=1]

define i32 @test(i32* %P1, i32* %P2, i32* %P3) nounwind {
        %L = load i32* @GLOBAL          ; <i32> [#uses=1]
        store i32 12, i32* %P2
        %Y = load i32* %P3              ; <i32> [#uses=1]
        %Z = sub i32 %Y, %L             ; <i32> [#uses=1]
        ret i32 %Z
}

