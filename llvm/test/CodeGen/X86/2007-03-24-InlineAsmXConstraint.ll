; RUN: llvm-as < %s | llc -march=x86 | grep {psrlw \$8, %xmm0}
target datalayout = "e-p:32:32"
target triple = "i686-apple-darwin9"

define void @test() {
        tail call void asm sideeffect "psrlw $0, %xmm0", "X,~{dirflag},~{fpsr},~{flags}"( i32 8 )
        ret void
}

