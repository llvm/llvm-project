; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s -check-prefix=PPC32
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s -check-prefix=PPC64

; PPC32: test
; PPC32: 4, 32751(3)
; PPC32: blr
; PPC64: test
; PPC64: 4, 32751(3)
; PPC64: blr
define void @test() nounwind {
	store i32 0, i32* inttoptr (i64 48725999 to i32*)
	ret void
}

; PPC32: test2
; PPC32: stw 4, 9028(3)
; PPC32: stw 4, 9024(3)
; PPC32: blr
; PPC64: test2
; PPC64: std 4, 9024(3)
; PPC64: blr
define void @test2() nounwind {
	store i64 0, i64* inttoptr (i64 74560 to i64*)
	ret void
}

