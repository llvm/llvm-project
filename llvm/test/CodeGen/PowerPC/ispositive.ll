; RUN: llvm-as < %s | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 | \
; RUN:   grep {srwi r3, r3, 31}

define i32 @test1(i32 %X) {
entry:
        icmp slt i32 %X, 0              ; <i1>:0 [#uses=1]
        zext i1 %0 to i32               ; <i32>:1 [#uses=1]
        ret i32 %1
}

