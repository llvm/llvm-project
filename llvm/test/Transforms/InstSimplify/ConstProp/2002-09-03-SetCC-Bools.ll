; SetCC on boolean values was not implemented!

; RUN: opt < %s -passes=instsimplify,dce -S | \
; RUN:   not grep set

define i1 @test1() {
        %A = icmp ule i1 true, false            ; <i1> [#uses=1]
        %B = icmp uge i1 true, false            ; <i1> [#uses=1]
        %C = icmp ult i1 false, true            ; <i1> [#uses=1]
        %D = icmp ugt i1 true, false            ; <i1> [#uses=1]
        %E = icmp eq i1 false, false            ; <i1> [#uses=1]
        %F = icmp ne i1 false, true             ; <i1> [#uses=1]
        %G = and i1 %A, %B              ; <i1> [#uses=1]
        %H = and i1 %C, %D              ; <i1> [#uses=1]
        %I = and i1 %E, %F              ; <i1> [#uses=1]
        %J = and i1 %G, %H              ; <i1> [#uses=1]
        %K = and i1 %I, %J              ; <i1> [#uses=1]
        ret i1 %K
}

