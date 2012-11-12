; RUN: llc < %s

        %struct..0anon = type { [100 x i32] }

define void @test() {
entry:
        %currfpu = alloca %struct..0anon, align 16              ; <%struct..0anon*> [#uses=2]
        %mxcsr = alloca %struct..0anon, align 16                ; <%struct..0anon*> [#uses=1]
        call void asm sideeffect "fnstenv $0", "=*m,~{dirflag},~{fpsr},~{flags}"( %struct..0anon* %currfpu )
        call void asm sideeffect "$0  $1", "=*m,*m,~{dirflag},~{fpsr},~{flags}"( %struct..0anon* %mxcsr, %struct..0anon* %currfpu )
        ret void
}

