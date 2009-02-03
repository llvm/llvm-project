; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {ret i1 true}
; rdar://5278853

define i1 @test(i32 %tmp468) {
        %tmp470 = udiv i32 %tmp468, 4           ; <i32> [#uses=2]
        %tmp475 = icmp ult i32 %tmp470, 1073741824              ; <i1> [#uses=1]
        ret i1 %tmp475
}

