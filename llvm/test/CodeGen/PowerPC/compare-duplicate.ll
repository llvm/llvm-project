; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu  | not grep slwi

define i32 @test(i32 %A, i32 %B) {
	%C = sub i32 %B, %A
	%D = icmp eq i32 %C, %A
	br i1 %D, label %T, label %F
T:
	ret i32 19123
F:
	ret i32 %C
}
