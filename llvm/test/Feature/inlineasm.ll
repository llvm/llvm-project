; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

module asm "this is an inline asm block"
module asm "this is another inline asm block"

define i32 @test() {
        %X = call i32 asm "tricky here $0, $1", "=r,r"( i32 4 )         ; <i32> [#uses=1]
        call void asm sideeffect "eieio", ""( )
        ret i32 %X
}

