; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    grep {ret i1 false}

define i1 @test(i1 %V) {
        %Y = icmp ult i1 %V, false              ; <i1> [#uses=1]
        ret i1 %Y
}

