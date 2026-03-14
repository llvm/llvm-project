; RUN: opt < %s -passes=instcombine -S | grep nounwind

define void @bar() {
entry:
        call void asm sideeffect "", "~{dirflag},~{fpsr},~{flags}"( )
        ret void
}
