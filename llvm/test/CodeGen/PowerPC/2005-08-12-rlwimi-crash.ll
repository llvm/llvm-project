; this should not crash the ppc backend

; RUN: llc -verify-machineinstrs < %s -march=ppc32


define i32 @test(i32 %j.0.0.i) {
        %tmp.85.i = and i32 %j.0.0.i, 7         ; <i32> [#uses=1]
        %tmp.161278.i = bitcast i32 %tmp.85.i to i32            ; <i32> [#uses=1]
        %tmp.5.i77.i = lshr i32 %tmp.161278.i, 3                ; <i32> [#uses=1]
        ret i32 %tmp.5.i77.i
}


