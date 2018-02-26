; RUN: llc < %s -mtriple=i686--
; PR1259

define void @test() {
        %tmp2 = call i32 asm "...", "=r,~{dirflag},~{fpsr},~{flags},~{dx},~{cx},~{ax}"( )
        unreachable
}
