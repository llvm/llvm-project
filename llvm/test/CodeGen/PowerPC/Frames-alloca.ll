; RUN: llc < %s -march=ppc32 -mtriple=powerpc-apple-darwin8 | FileCheck %s -check-prefix=CHECK-PPC32
; RUN: llc < %s -march=ppc64 -mtriple=powerpc-apple-darwin8 | FileCheck %s -check-prefix=CHECK-PPC64
; RUN: llc < %s -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | FileCheck %s -check-prefix=CHECK-PPC32-NOFP
; RUN: llc < %s -march=ppc64 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | FileCheck %s -check-prefix=CHECK-PPC64-NOFP
; RUN: llc < %s -march=ppc32 -mtriple=powerpc-apple-darwin8 | FileCheck %s -check-prefix=CHECK-PPC32
; RUN: llc < %s -march=ppc32 -mtriple=powerpc-apple-darwin8 | FileCheck %s -check-prefix=CHECK-PPC32-RS
; RUN: llc < %s -march=ppc32 -mtriple=powerpc-apple-darwin8 -disable-fp-elim | FileCheck %s -check-prefix=CHECK-PPC32-RS-NOFP

; CHECK-PPC32: stw r31, -4(r1)
; CHECK-PPC32: lwz r1, 0(r1)
; CHECK-PPC32: lwz r31, -4(r1)
; CHECK-PPC32-NOFP: stw r31, -4(r1)
; CHECK-PPC32-NOFP: lwz r1, 0(r1)
; CHECK-PPC32-NOFP: lwz r31, -4(r1)
; CHECK-PPC32-RS: stwu r1, -48(r1)
; CHECK-PPC32-RS-NOFP: stwu r1, -48(r1)

; CHECK-PPC64: std r31, -8(r1)
; CHECK-PPC64: stdu r1, -64(r1)
; CHECK-PPC64: ld r1, 0(r1)
; CHECK-PPC64: ld r31, -8(r1)
; CHECK-PPC64-NOFP: std r31, -8(r1)
; CHECK-PPC64-NOFP: stdu r1, -64(r1)
; CHECK-PPC64-NOFP: ld r1, 0(r1)
; CHECK-PPC64-NOFP: ld r31, -8(r1)

define i32* @f1(i32 %n) nounwind {
	%tmp = alloca i32, i32 %n		; <i32*> [#uses=1]
	ret i32* %tmp
}
