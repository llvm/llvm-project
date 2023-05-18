; RUN: llc < %s

define i64 @test_imm(i64 %X) {
        %Y = ashr i64 %X, 17            ; <i64> [#uses=1]
        ret i64 %Y
}

define i64 @test_variable(i64 %X, i8 %Amt) {
        %shift.upgrd.1 = zext i8 %Amt to i64            ; <i64> [#uses=1]
        %Y = ashr i64 %X, %shift.upgrd.1                ; <i64> [#uses=1]
        ret i64 %Y
}
