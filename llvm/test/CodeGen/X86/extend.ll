; RUN: llc < %s -march=x86 -x86-asm-syntax=intel | grep movzx | count 1
; RUN: llc < %s -march=x86 -x86-asm-syntax=intel | grep movsx | count 1

@G1 = internal global i8 0              ; <i8*> [#uses=1]
@G2 = internal global i8 0              ; <i8*> [#uses=1]

define i16 @test1() {
        %tmp.0 = load i8, i8* @G1           ; <i8> [#uses=1]
        %tmp.3 = zext i8 %tmp.0 to i16          ; <i16> [#uses=1]
        ret i16 %tmp.3
}

define i16 @test2() {
        %tmp.0 = load i8, i8* @G2           ; <i8> [#uses=1]
        %tmp.3 = sext i8 %tmp.0 to i16          ; <i16> [#uses=1]
        ret i16 %tmp.3
}

