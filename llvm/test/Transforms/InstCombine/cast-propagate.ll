; RUN: llvm-as < %s | opt -instcombine -mem2reg | llvm-dis | \
; RUN:    not grep load

define i32 @test1(i32* %P) {
        %A = alloca i32         ; <i32*> [#uses=2]
        store i32 123, i32* %A
        ; Cast the result of the load not the source
        %Q = bitcast i32* %A to i32*            ; <i32*> [#uses=1]
        %V = load i32* %Q               ; <i32> [#uses=1]
        ret i32 %V
}
