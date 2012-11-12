; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mattr=+thumb2 | FileCheck %s

@b = external global i64*

define i64 @t(i64 %a) nounwind readonly {
entry:
; CHECK: ldrd
; CHECK: umull
	%0 = load i64** @b, align 4
	%1 = load i64* %0, align 4
	%2 = mul i64 %1, %a
	ret i64 %2
}
