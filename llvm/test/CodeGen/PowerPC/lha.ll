; RUN: llc -verify-machineinstrs < %s -march=ppc32 | grep lha

define i32 @test(i16* %a) {
        %tmp.1 = load i16, i16* %a           ; <i16> [#uses=1]
        %tmp.2 = sext i16 %tmp.1 to i32         ; <i32> [#uses=1]
        ret i32 %tmp.2
}

