; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-apple-darwin8 | FileCheck %s

define i32 @test(i32 %x) nounwind {
; CHECK-LABEL: @test
; CHECK: cmpwi r3, -1

        %c = icmp eq i32 %x, -1
	br i1 %c, label %T, label %F
T:
	%A = call i32 @test(i32 123)
	%B = add i32 %A, 43
	ret i32 %B
F:
	%G = add i32 %x, 1234
	ret i32 %G
}
