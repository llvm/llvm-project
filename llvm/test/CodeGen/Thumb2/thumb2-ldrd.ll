; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mattr=+thumb2 | FileCheck %s

@b = external global ptr

define i64 @t(i64 %a) nounwind readonly {
entry:
; CHECK: ldrd
; CHECK: umull
	%0 = load ptr, ptr @b, align 4
	%1 = load i64, ptr %0, align 4
	%2 = mul i64 %1, %a
	ret i64 %2
}
