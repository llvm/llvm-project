; This test should not contain a sign extend
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | not grep extsb

define i32 @test(i32 %mode.0.i.0) {
        %tmp.79 = trunc i32 %mode.0.i.0 to i8           ; <i8> [#uses=1]
        %tmp.80 = sext i8 %tmp.79 to i32                ; <i32> [#uses=1]
        %tmp.81 = shl i32 %tmp.80, 24           ; <i32> [#uses=1]
        ret i32 %tmp.81
}

define i32 @test2(i32 %mode.0.i.0) {
        %tmp.79 = trunc i32 %mode.0.i.0 to i8           ; <i8> [#uses=1]
        %tmp.80 = sext i8 %tmp.79 to i32                ; <i32> [#uses=1]
        %tmp.81 = shl i32 %tmp.80, 16           ; <i32> [#uses=1]
        %tmp.82 = and i32 %tmp.81, 16711680             ; <i32> [#uses=1]
        ret i32 %tmp.82
}

