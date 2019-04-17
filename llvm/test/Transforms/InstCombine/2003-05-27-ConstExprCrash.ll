; RUN: opt < %s -instcombine -disable-output

@X = global i32 5               ; <i32*> [#uses=1]

define i64 @test() {
        %C = add i64 1, 2               ; <i64> [#uses=1]
        %V = add i64 ptrtoint (i32* @X to i64), %C              ; <i64> [#uses=1]
        ret i64 %V
}

