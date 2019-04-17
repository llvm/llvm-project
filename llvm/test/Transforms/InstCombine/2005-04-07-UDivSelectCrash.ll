; RUN: opt < %s -instcombine -disable-output

define i32 @test(i1 %C, i32 %tmp.15) {
        %tmp.16 = select i1 %C, i32 8, i32 1            ; <i32> [#uses=1]
        %tmp.18 = udiv i32 %tmp.15, %tmp.16             ; <i32> [#uses=1]
        ret i32 %tmp.18
}

