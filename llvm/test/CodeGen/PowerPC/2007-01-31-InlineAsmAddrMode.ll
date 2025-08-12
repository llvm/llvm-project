; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32--
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64--

; Test two things: 1) that a frameidx can be rewritten in an inline asm
; 2) that inline asms can handle reg+imm addr modes.

	%struct.A = type { i32, i32 }


define void @test1() {
entry:
	%Out = alloca %struct.A, align 4		; <ptr> [#uses=1]
	%tmp2 = getelementptr %struct.A, ptr %Out, i32 0, i32 1
	%tmp5 = call i32 asm "lbz $0, $1", "=r,m"(ptr %tmp2 )
	ret void
}

define void @test2() {
entry:
	%Out = alloca %struct.A, align 4		; <ptr> [#uses=1]
	%tmp2 = getelementptr %struct.A, ptr %Out, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp5 = call i32 asm "lwbrx $0, $2, $1", "=r,r,bO,m"( ptr null, i32 0, ptr %tmp2 )		; <i32> [#uses=0]
	ret void
}
