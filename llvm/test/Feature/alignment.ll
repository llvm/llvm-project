; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

@X = global i32 4, align 16             ; <i32*> [#uses=0]

define i32* @test() align 32 {
        %X = alloca i32, align 4                ; <i32*> [#uses=1]
        %Y = alloca i32, i32 42, align 16               ; <i32*> [#uses=0]
        %Z = alloca i32         ; <i32*> [#uses=0]
        ret i32* %X
}
define void @test3() alignstack(16) {
        ret void
}

