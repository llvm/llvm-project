; Tests to make sure elimination of casts is working correctly
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:   notcast {} {%c1.*}

define i64 @test_sext_zext(i16 %A) {
        %c1 = zext i16 %A to i32                ; <i32> [#uses=1]
        %c2 = sext i32 %c1 to i64               ; <i64> [#uses=1]
        ret i64 %c2
}

