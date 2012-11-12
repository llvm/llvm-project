; RUN: llc < %s -march=x86 | FileCheck %s
; This should not load or store the top part of *P.

define void @test(i64* %P) nounwind  {
; CHECK: test:
; CHECK: movl 4(%esp), %[[REGISTER:.*]]
; CHECK-NOT: 4(%[[REGISTER]])
; CHECK: ret
	%tmp1 = load i64* %P, align 8		; <i64> [#uses=1]
	%tmp2 = xor i64 %tmp1, 1		; <i64> [#uses=1]
	store i64 %tmp2, i64* %P, align 8
	ret void
}

