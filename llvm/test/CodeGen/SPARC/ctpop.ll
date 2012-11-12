; RUN: llc < %s -march=sparc -mattr=-v9 | not grep popc
; RUN: llc < %s -march=sparcv9 -mattr=v9 | grep popc

declare i32 @llvm.ctpop.i32(i32)

define i32 @test(i32 %X) {
        %Y = call i32 @llvm.ctpop.i32( i32 %X )         ; <i32> [#uses=1]
        ret i32 %Y
}

