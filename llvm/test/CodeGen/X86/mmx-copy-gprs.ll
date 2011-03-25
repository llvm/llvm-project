; RUN: llc < %s -mtriple=x86_64-linux   | FileCheck %s -check-prefix=X64
; RUN: llc < %s -mtriple=x86_64-win32   | FileCheck %s -check-prefix=X64
; X64: movq ({{%rsi|%rdx}}), %rax
; RUN: llc < %s -march=x86 -mattr=-sse2 | FileCheck %s -check-prefix=X32
; X32: movl 4(%eax),
; RUN: llc < %s -march=x86 -mattr=+sse2 | FileCheck %s -check-prefix=XMM
; XMM: movsd (%eax),

; This test should use GPRs to copy the mmx value, not MMX regs.  Using mmx regs,
; increases the places that need to use emms.

; rdar://5741668

define void @foo(<1 x i64>* %x, <1 x i64>* %y) nounwind  {
entry:
	%tmp1 = load <1 x i64>* %y, align 8		; <<1 x i64>> [#uses=1]
	store <1 x i64> %tmp1, <1 x i64>* %x, align 8
	ret void
}
