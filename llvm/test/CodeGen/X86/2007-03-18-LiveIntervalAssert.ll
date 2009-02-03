; RUN: llvm-as < %s | llc -march=x86
; PR1259

define void @test() {
        %tmp2 = call i32 asm "...", "=r,~{dirflag},~{fpsr},~{flags},~{dx},~{cx},~{ax}"( )
        unreachable
}
