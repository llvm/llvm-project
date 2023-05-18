; RUN: llc < %s -mtriple=i686-- -no-integrated-as
; RUN: llc < %s -mtriple=x86_64-- -no-integrated-as

; from gcc.c-torture/compile/920520-1.c

define i32 @g() nounwind {
entry:
	call void asm sideeffect "$0", "r"(double 1.500000e+00) nounwind
	ret i32 0
}

