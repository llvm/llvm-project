; RUN: llc < %s -mtriple=x86_64-linux   | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win32   | FileCheck %s
; RUN: llc < %s -mtriple=i686-- -mattr=-sse2 | FileCheck %s
; RUN: llc < %s -mtriple=i686-- -mattr=+sse2 | FileCheck %s

; This test should use GPRs to copy the mmx value, not MMX regs.  Using mmx regs,
; increases the places that need to use emms.
; CHECK-NOT: %mm
; CHECK-NOT: emms
; rdar://5741668

define void @foo(ptr %x, ptr %y) nounwind  {
entry:
	%tmp1 = load <1 x i64>, ptr %y, align 8		; <<1 x i64>> [#uses=1]
	store <1 x i64> %tmp1, ptr %x, align 8
	ret void
}
