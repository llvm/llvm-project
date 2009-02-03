; RUN: llvm-as < %s | llc -march=x86
; PR825

define i64 @test() {
	%tmp.i5 = call i64 asm sideeffect "rdtsc", "=A,~{dirflag},~{fpsr},~{flags}"( )		; <i64> [#uses=1]
	ret i64 %tmp.i5
}

