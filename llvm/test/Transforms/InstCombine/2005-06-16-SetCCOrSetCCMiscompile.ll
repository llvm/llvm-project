; RUN: opt < %s -instcombine -S | \
; RUN:   grep "ret i1 true"
; PR586

@g_07918478 = external global i32               ; <i32*> [#uses=1]

define i1 @test() {
        %tmp.0 = load i32* @g_07918478          ; <i32> [#uses=2]
        %tmp.1 = icmp ne i32 %tmp.0, 0          ; <i1> [#uses=1]
        %tmp.4 = icmp ult i32 %tmp.0, 4111              ; <i1> [#uses=1]
        %bothcond = or i1 %tmp.1, %tmp.4                ; <i1> [#uses=1]
        ret i1 %bothcond
}

