; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | grep lha

define i32 @test(ptr %a) {
        %tmp.1 = load i16, ptr %a           ; <i16> [#uses=1]
        %tmp.2 = sext i16 %tmp.1 to i32         ; <i32> [#uses=1]
        ret i32 %tmp.2
}

