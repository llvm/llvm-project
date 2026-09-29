; RUN: llc -no-integrated-as < %s

; Test that we can have an "X" output constraint.

define void @test(ptr %t) {
        call void asm sideeffect "foo $0", "=*X,~{dirflag},~{fpsr},~{flags},~{memory}"( ptr elementtype( i16) %t )
        ret void
}
