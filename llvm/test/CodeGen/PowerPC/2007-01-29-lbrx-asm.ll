; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32--
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64--

define i16 @test(ptr %d1, ptr %d2) {
	%tmp237 = call i16 asm "lhbrx $0, $2, $1", "=r,r,bO,m"( ptr %d1, i32 0, ptr %d2 )		; <i16> [#uses=1]
	ret i16 %tmp237
}
