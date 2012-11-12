; RUN: llc < %s -march=x86
; RUN: llc < %s -march=x86-64

define void @test(i64 %x) nounwind {
entry:
	tail call void asm sideeffect "ASM: $0", "r,~{dirflag},~{fpsr},~{flags}"(i64 %x) nounwind
	ret void
}

