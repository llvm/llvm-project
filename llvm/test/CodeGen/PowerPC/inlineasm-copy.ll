; RUN: llvm-as < %s | llc -march=ppc32 | not grep mr

define i32 @test(i32 %Y, i32 %X) {
entry:
        %tmp = tail call i32 asm "foo $0", "=r"( )              ; <i32> [#uses=1]
        ret i32 %tmp
}

define i32 @test2(i32 %Y, i32 %X) {
entry:
        %tmp1 = tail call i32 asm "foo $0, $1", "=r,r"( i32 %X )                ; <i32> [#uses=1]
        ret i32 %tmp1
}

