; Neither of these functions should contain algebraic right shifts
; RUN: llvm-as < %s | llc -march=ppc32 | not grep srawi 

define i32 @test1(i32 %mode.0.i.0) {
        %tmp.79 = bitcast i32 %mode.0.i.0 to i32                ; <i32> [#uses=1]
        %tmp.80 = ashr i32 %tmp.79, 15          ; <i32> [#uses=1]
        %tmp.81 = and i32 %tmp.80, 24           ; <i32> [#uses=1]
        ret i32 %tmp.81
}

define i32 @test2(i32 %mode.0.i.0) {
        %tmp.79 = bitcast i32 %mode.0.i.0 to i32                ; <i32> [#uses=1]
        %tmp.80 = ashr i32 %tmp.79, 15          ; <i32> [#uses=1]
        %tmp.81 = lshr i32 %mode.0.i.0, 16              ; <i32> [#uses=1]
        %tmp.82 = bitcast i32 %tmp.81 to i32            ; <i32> [#uses=1]
        %tmp.83 = and i32 %tmp.80, %tmp.82              ; <i32> [#uses=1]
        ret i32 %tmp.83
}

define i32 @test3(i32 %specbits.6.1) {
        %tmp.2540 = ashr i32 %specbits.6.1, 11          ; <i32> [#uses=1]
        %tmp.2541 = bitcast i32 %tmp.2540 to i32                ; <i32> [#uses=1]
        %tmp.2542 = shl i32 %tmp.2541, 13               ; <i32> [#uses=1]
        %tmp.2543 = and i32 %tmp.2542, 8192             ; <i32> [#uses=1]
        ret i32 %tmp.2543
}

