; RUN: llc -verify-machineinstrs < %s -march=ppc32 | FileCheck %s -check-prefix=PPC32-NOFP
; RUN: llc -verify-machineinstrs < %s -march=ppc32 -disable-fp-elim | FileCheck %s -check-prefix=PPC32-FP

; RUN: llc -verify-machineinstrs < %s -march=ppc64 | FileCheck %s -check-prefix=PPC64-NOFP
; RUN: llc -verify-machineinstrs < %s -march=ppc64 -disable-fp-elim | FileCheck %s -check-prefix=PPC64-FP


target triple = "powerpc-apple-darwin8"

define i32* @f1() nounwind {
        %tmp = alloca i32, i32 8191             ; <i32*> [#uses=1]
        ret i32* %tmp
}

; PPC32-NOFP: _f1:
; PPC32-NOFP: 	lis r0, -1
; PPC32-NOFP: 	ori r0, r0, 32736
; PPC32-NOFP: 	stwux r1, r1, r0
; PPC32-NOFP: 	addi r3, r1, 36
; PPC32-NOFP: 	lwz r1, 0(r1)
; PPC32-NOFP: 	blr 


; PPC32-FP: _f1:
; PPC32-FP:	lis r0, -1
; PPC32-FP:	stw r31, -4(r1)
; PPC32-FP:	ori r0, r0, 32736
; PPC32-FP:	stwux r1, r1, r0
; PPC32-FP:	mr r31, r1
; PPC32-FP:	addi r3, r31, 32
; PPC32-FP:	lwz r1, 0(r1)
; PPC32-FP:	lwz r31, -4(r1)
; PPC32-FP:	blr 


; PPC64-NOFP: _f1:
; PPC64-NOFP: 	lis r0, -1
; PPC64-NOFP: 	ori r0, r0, 32720
; PPC64-NOFP: 	stdux r1, r1, r0
; PPC64-NOFP: 	addi r3, r1, 52
; PPC64-NOFP: 	ld r1, 0(r1)
; PPC64-NOFP: 	blr 


; PPC64-FP: _f1:
; PPC64-FP:	lis r0, -1
; PPC64-FP:	std r31, -8(r1)
; PPC64-FP:	ori r0, r0, 32704
; PPC64-FP:	stdux r1, r1, r0
; PPC64-FP:	mr r31, r1
; PPC64-FP:	addi r3, r31, 60
; PPC64-FP:	ld r1, 0(r1)
; PPC64-FP:	ld r31, -8(r1)
; PPC64-FP:	blr 
