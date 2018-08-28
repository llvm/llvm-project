; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s -check-prefix=PPC32-NOFP
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu -disable-fp-elim | FileCheck %s -check-prefix=PPC32-FP

; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s -check-prefix=PPC64-NOFP
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -disable-fp-elim | FileCheck %s -check-prefix=PPC64-FP

define i32* @f1() nounwind {
        %tmp = alloca i32, i32 8191             ; <i32*> [#uses=1]
        ret i32* %tmp
}

; PPC32-NOFP: f1:
; PPC32-NOFP: 	lis 0, -1
; PPC32-NOFP: 	ori 0, 0, 32752
; PPC32-NOFP: 	stwux 1, 1, 0
; PPC32-NOFP-DAG: 	addi 3, 1, 20
; PPC32-NOFP-DAG: 	lwz 31, 0(1)
; PPC32-NOFP:   mr 1, 31
; PPC32-NOFP:   mr 31, 0
; PPC32-NOFP: 	blr

; PPC32-FP: lis 0, -1
; PPC32-FP: ori 0, 0, 32752
; PPC32-FP: stwux 1, 1, 0
; PPC32-FP: subf 0, 0, 1
; PPC32-FP: addic 0, 0, -4
; PPC32-FP: stwx 31, 0, 0
; PPC32-FP: mr 31, 1
; PPC32-FP: addi 3, 31, 16
; PPC32-FP: lwz 31, 0(1)
; PPC32-FP: lwz 0, -4(31)
; PPC32-FP: mr 1, 31
; PPC32-FP: mr 31, 0
; PPC32-FP: blr

; PPC64-NOFP: f1:
; PPC64-NOFP: 	lis 0, -1
; PPC64-NOFP: 	ori 0, 0, 32720
; PPC64-NOFP: 	stdux 1, 1, 0
; PPC64-NOFP: 	addi 3, 1, 52
; PPC64-NOFP: 	ld 1, 0(1)
; PPC64-NOFP: 	blr

; PPC64-FP: f1:
; PPC64-FP:	lis 0, -1
; PPC64-FP:	ori 0, 0, 32704
; PPC64-FP:	std 31, -8(1)
; PPC64-FP:	stdux 1, 1, 0
; PPC64-FP:	mr 31, 1
; PPC64-FP:	addi 3, 31, 60
; PPC64-FP:	ld 1, 0(1)
; PPC64-FP:	ld 31, -8(1)
; PPC64-FP:	blr
