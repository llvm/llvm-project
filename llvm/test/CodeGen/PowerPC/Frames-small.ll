; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s -check-prefix=PPC32-FP
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu -disable-fp-elim | FileCheck %s -check-prefix=PPC32-NOFP
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s -check-prefix=PPC64-FP
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -disable-fp-elim | FileCheck %s -check-prefix=PPC64-NOFP

;PPC32-FP: f1:
;PPC32-FP: stwu 1, -16400(1)
;PPC32-FP: addi 3, 1, 20
;PPC32-FP: addi 1, 1, 16400
;PPC32-FP: blr

;PPC32-NOFP: f1:
;PPC32-NOFP: stwu 1, -16400(1)
;PPC32-NOFP: stw 31, 16396(1)
;PPC32-NOFP: lwz 31, 16396(1)
;PPC32-NOFP: addi 1, 1, 16400
;PPC32-NOFP: blr

;PPC64-FP: f1:
;PPC64-FP: stdu 1, -16432(1)
;PPC64-FP: addi 1, 1, 16432
;PPC64-FP: blr

;PPC64-NOFP: f1:
;PPC64-NOFP: std 31, -8(1)
;PPC64-NOFP: stdu 1, -16448(1)
;PPC64-NOFP: addi 1, 1, 16448
;PPC64-NOFP: ld 31, -8(1)
;PPC64-NOFP: blr


define i32* @f1() {
        %tmp = alloca i32, i32 4095             ; <i32*> [#uses=1]
        ret i32* %tmp
}

