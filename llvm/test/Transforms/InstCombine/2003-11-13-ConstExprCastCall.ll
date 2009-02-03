; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep call | notcast

declare void @free(i8*)

define void @test(i32* %X) {
        call void (...)* bitcast (void (i8*)* @free to void (...)*)( i32* %X )          ; <i32>:1 [#uses=0]
        ret void
}
