; RUN: opt < %s -instcombine -disable-output

define i32 @test() {
        %tmp203 = icmp eq i32 1, 2              ; <i1> [#uses=1]
        %tmp203.upgrd.1 = zext i1 %tmp203 to i32                ; <i32> [#uses=1]
        ret i32 %tmp203.upgrd.1
}

