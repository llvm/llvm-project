; RUN: llc < %s -march=x86 -x86-asm-syntax=intel | \
; RUN:   grep "sh[lr]d" | count 5

define i64 @test1(i64 %X, i8 %C) {
        %shift.upgrd.1 = zext i8 %C to i64              ; <i64> [#uses=1]
        %Y = shl i64 %X, %shift.upgrd.1         ; <i64> [#uses=1]
        ret i64 %Y
}

define i64 @test2(i64 %X, i8 %C) {
        %shift.upgrd.2 = zext i8 %C to i64              ; <i64> [#uses=1]
        %Y = ashr i64 %X, %shift.upgrd.2                ; <i64> [#uses=1]
        ret i64 %Y
}

define i64 @test3(i64 %X, i8 %C) {
        %shift.upgrd.3 = zext i8 %C to i64              ; <i64> [#uses=1]
        %Y = lshr i64 %X, %shift.upgrd.3                ; <i64> [#uses=1]
        ret i64 %Y
}

define i32 @test4(i32 %A, i32 %B, i8 %C) {
        %shift.upgrd.4 = zext i8 %C to i32              ; <i32> [#uses=1]
        %X = shl i32 %A, %shift.upgrd.4         ; <i32> [#uses=1]
        %Cv = sub i8 32, %C             ; <i8> [#uses=1]
        %shift.upgrd.5 = zext i8 %Cv to i32             ; <i32> [#uses=1]
        %Y = lshr i32 %B, %shift.upgrd.5                ; <i32> [#uses=1]
        %Z = or i32 %Y, %X              ; <i32> [#uses=1]
        ret i32 %Z
}

define i16 @test5(i16 %A, i16 %B, i8 %C) {
        %shift.upgrd.6 = zext i8 %C to i16              ; <i16> [#uses=1]
        %X = shl i16 %A, %shift.upgrd.6         ; <i16> [#uses=1]
        %Cv = sub i8 16, %C             ; <i8> [#uses=1]
        %shift.upgrd.7 = zext i8 %Cv to i16             ; <i16> [#uses=1]
        %Y = lshr i16 %B, %shift.upgrd.7                ; <i16> [#uses=1]
        %Z = or i16 %Y, %X              ; <i16> [#uses=1]
        ret i16 %Z
}

