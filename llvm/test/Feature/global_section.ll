; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

@X = global i32 4, section "foo", align 16              ; <i32*> [#uses=0]

define void @test() section "bar" {
        ret void
}

