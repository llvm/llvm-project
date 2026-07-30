; RUN: opt < %s -passes=instcombine -S | grep 4294967295

define i64 @test(i64 %Val) {
        %tmp.3 = trunc i64 %Val to i32          ; <i32> [#uses=1]
        %tmp.8 = zext i32 %tmp.3 to i64         ; <i64> [#uses=1]
        ret i64 %tmp.8
}

