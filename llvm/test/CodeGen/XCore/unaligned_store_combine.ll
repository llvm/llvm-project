; RUN: llc < %s -march=xcore | FileCheck %s

; Unaligned load / store pair. Should be combined into a memmove
; of size 8
define void @f(i64* %dst, i64* %src) nounwind {
entry:
; CHECK: f:
; CHECK: ldc r2, 8
; CHECK: bl memmove
	%0 = load i64* %src, align 1
	store i64 %0, i64* %dst, align 1
	ret void
}
