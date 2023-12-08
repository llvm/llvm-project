; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

@X = global i32 4, align 16             ; <ptr> [#uses=0]

define ptr @test() align 32 {
        %X = alloca i32, align 4                ; <ptr> [#uses=1]
        %Y = alloca i32, i32 42, align 16               ; <ptr> [#uses=0]
        %Z = alloca i32         ; <ptr> [#uses=0]
        ret ptr %X
}
define void @test3() alignstack(16) {
        ret void
}

