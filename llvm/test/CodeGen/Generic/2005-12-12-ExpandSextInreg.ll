; RUN: llc < %s

define i64 @test(i64 %A) {
        %B = trunc i64 %A to i8         ; <i8> [#uses=1]
        %C = sext i8 %B to i64          ; <i64> [#uses=1]
        ret i64 %C
}
