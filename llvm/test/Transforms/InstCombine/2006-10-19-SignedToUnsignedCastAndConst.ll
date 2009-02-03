; This test case is reduced from llvmAsmParser.cpp
; The optimizer should not remove the cast here.
; RUN: llvm-as %s -o - | opt -instcombine | llvm-dis | \
; RUN:    grep sext.*i32


define i1 @test(i16 %X) {
        %A = sext i16 %X to i32         ; <i32> [#uses=1]
        %B = icmp ugt i32 %A, 1330              ; <i1> [#uses=1]
        ret i1 %B
}

