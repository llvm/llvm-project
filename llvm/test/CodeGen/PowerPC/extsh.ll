; This should turn into a single extsh
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | grep extsh | count 1
define i32 @test(i32 %X) {
        %tmp.81 = shl i32 %X, 16                ; <i32> [#uses=1]
        %tmp.82 = ashr i32 %tmp.81, 16          ; <i32> [#uses=1]
        ret i32 %tmp.82
}

