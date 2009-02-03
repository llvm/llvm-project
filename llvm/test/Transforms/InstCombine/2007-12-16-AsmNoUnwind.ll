; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep nounwind

define void @bar() {
entry:
        call void asm sideeffect "", "~{dirflag},~{fpsr},~{flags}"( )
        ret void
}
