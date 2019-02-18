; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s -check-prefix=CHECK-PPC32
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s -check-prefix=CHECK-PPC64
; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu -disable-fp-elim | FileCheck %s -check-prefix=CHECK-PPC32-NOFP
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -disable-fp-elim | FileCheck %s -check-prefix=CHECK-PPC64-NOFP
; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s -check-prefix=CHECK-PPC32

; CHECK-PPC32: stwu 1, -32(1)
; CHECK-PPC32: stw 31, 28(1)
; CHECK-PPC32: lwz 31, 0(1)
; CHECK-PPC32-NOFP: stwu 1, -32(1)
; CHECK-PPC32-NOFP: stw 31, 28(1)
; CHECK-PPC32-NOFP: lwz 31, 0(1)

; CHECK-PPC64: std 31, -8(1)
; CHECK-PPC64: stdu 1, -64(1)
; CHECK-PPC64: ld 1, 0(1)
; CHECK-PPC64: ld 31, -8(1)
; CHECK-PPC64-NOFP: std 31, -8(1)
; CHECK-PPC64-NOFP: stdu 1, -64(1)
; CHECK-PPC64-NOFP: ld 1, 0(1)
; CHECK-PPC64-NOFP: ld 31, -8(1)

define i32* @f1(i32 %n) nounwind {
	%tmp = alloca i32, i32 %n		; <i32*> [#uses=1]
	ret i32* %tmp
}
