; RUN: opt < %s -passes=instcombine -S | \
; RUN:   grep "ret i1 false"
define i1 @test() {
        %X = trunc i32 320 to i1                ; <i1> [#uses=1]
        ret i1 %X
}

