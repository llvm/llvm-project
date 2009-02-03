; RUN: llvm-as < %s | llc | grep 68719476738

define void @test() {
entry:
        tail call void asm sideeffect "/* result: ${0:c} */", "i,~{dirflag},~{fpsr},~{flags}"( i64 68719476738 )
        ret void
}

