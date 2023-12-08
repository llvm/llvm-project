; RUN: opt < %s -passes=instcombine -disable-output

@X = global i32 5               ; <ptr> [#uses=1]

define i64 @test() {
        %C = add i64 1, 2               ; <i64> [#uses=1]
        %V = add i64 ptrtoint (ptr @X to i64), %C              ; <i64> [#uses=1]
        ret i64 %V
}

