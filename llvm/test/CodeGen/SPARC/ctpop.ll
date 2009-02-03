; RUN: llvm-as < %s | \
; RUN:   llc -march=sparc -mattr=v9 -enable-sparc-v9-insts
; RUN: llvm-as < %s | llc -march=sparc -mattr=-v9 | \
; RUN:   not grep popc
; RUN: llvm-as < %s | \
; RUN:   llc -march=sparc -mattr=v9 -enable-sparc-v9-insts | grep popc

declare i32 @llvm.ctpop.i32(i32)

define i32 @test(i32 %X) {
        %Y = call i32 @llvm.ctpop.i32( i32 %X )         ; <i32> [#uses=1]
        ret i32 %Y
}

