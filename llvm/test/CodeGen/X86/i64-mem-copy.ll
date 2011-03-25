; RUN: llc < %s -mtriple=x86_64-linux   | FileCheck %s -check-prefix=X64
; RUN: llc < %s -mtriple=x86_64-win32   | FileCheck %s -check-prefix=X64
; X64: movq ({{%rsi|%rdx}}), %r

; RUN: llc < %s -march=x86 -mattr=+sse2 | FileCheck %s -check-prefix=X32
; X32: movsd (%eax), %xmm

; Uses movsd to load / store i64 values if sse2 is available.

; rdar://6659858

define void @foo(i64* %x, i64* %y) nounwind  {
entry:
	%tmp1 = load i64* %y, align 8		; <i64> [#uses=1]
	store i64 %tmp1, i64* %x, align 8
	ret void
}
