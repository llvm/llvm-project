; RUN: llc < %s -mtriple=i686--

define i1 @test(i1 %C, i1 %D, i32 %X, i32 %Y) {
        %E = icmp slt i32 %X, %Y                ; <i1> [#uses=1]
        %F = select i1 %C, i1 %D, i1 %E         ; <i1> [#uses=1]
        ret i1 %F
}

