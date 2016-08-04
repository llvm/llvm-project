; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu -mcpu=g5 | FileCheck %s

define void @test(float %F, i8* %P) {
	%I = fptosi float %F to i32
	%X = trunc i32 %I to i8
	store i8 %X, i8* %P
	ret void
; CHECK: fctiwz 0, 1
; CHECK: stfiwx 0, 0, 4
; CHECK: lwz 4, 12(1)
; CHECK: stb 4, 0(3)
; CHECK: blr
}

