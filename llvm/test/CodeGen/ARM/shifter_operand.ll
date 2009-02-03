; RUN: llvm-as < %s | llc -march=arm | grep add | grep lsl
; RUN: llvm-as < %s | llc -march=arm | grep bic | grep asr


define i32 @test1(i32 %X, i32 %Y, i8 %sh) {
        %shift.upgrd.1 = zext i8 %sh to i32             ; <i32> [#uses=1]
        %A = shl i32 %Y, %shift.upgrd.1         ; <i32> [#uses=1]
        %B = add i32 %X, %A             ; <i32> [#uses=1]
        ret i32 %B
}

define i32 @test2(i32 %X, i32 %Y, i8 %sh) {
        %shift.upgrd.2 = zext i8 %sh to i32             ; <i32> [#uses=1]
        %A = ashr i32 %Y, %shift.upgrd.2                ; <i32> [#uses=1]
        %B = xor i32 %A, -1             ; <i32> [#uses=1]
        %C = and i32 %X, %B             ; <i32> [#uses=1]
        ret i32 %C
}
