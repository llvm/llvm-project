; RUN: llc < %s -march=xcore | FileCheck %s

; Unaligned load / store pair. Should be combined into a memmove
; of size 8
define void @f(ptr %dst, ptr %src) nounwind {
entry:
; CHECK-LABEL: f:
; CHECK: ldc r2, 8
; CHECK: bl memmove
	%0 = load i64, ptr %src, align 1
	store i64 %0, ptr %dst, align 1
	ret void
}
