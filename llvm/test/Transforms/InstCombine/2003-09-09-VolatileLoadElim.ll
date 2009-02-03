; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep load

define void @test(i32* %P) {
        ; Dead but not deletable!
        %X = volatile load i32* %P              ; <i32> [#uses=0]
        ret void
}
