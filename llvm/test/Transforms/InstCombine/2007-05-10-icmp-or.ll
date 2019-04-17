; RUN: opt < %s -instcombine -disable-output
define i1 @test(i32 %tmp9) {
        %tmp20 = icmp ugt i32 %tmp9, 255                ; <i1> [#uses=1]
        %tmp11.not = icmp sgt i32 %tmp9, 255            ; <i1> [#uses=1]
        %bothcond = or i1 %tmp20, %tmp11.not            ; <i1> [#uses=1]
        ret i1 %bothcond
}

