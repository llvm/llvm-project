; RUN: llvm-as < %s | llc -march=x86 | grep {mov %gs:72, %eax}
target datalayout = "e-p:32:32"
target triple = "i686-apple-darwin9"

define void @test() {
	%tmp1 = tail call i32* asm sideeffect "mov %gs:${1:P}, $0", "=r,i,~{dirflag},~{fpsr},~{flags}"( i32 72 )		; <%struct._pthread*> [#uses=1]
	ret void
}


