; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu | \
; RUN: FileCheck %s -check-prefix=PPC32-LINUX-NOFP

; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu \
; RUN: -frame-pointer=all | FileCheck %s -check-prefix=PPC32-LINUX-FP

; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu | \
; RUN: FileCheck %s -check-prefix=PPC64-LINUX-NOFP

; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu \
; RUN: -frame-pointer=all | FileCheck %s -check-prefix=PPC64-LINUX-FP

; RUN: llc -verify-machineinstrs < %s -mcpu=pwr4 -mattr=-altivec \
; RUN: -mtriple=powerpc-ibm-aix-xcoff | FileCheck %s \
; RUN: -check-prefix=PPC32-AIX-NOFP

; RUN: llc -verify-machineinstrs < %s -mcpu=pwr4 -mattr=-altivec \
; RUN: -mtriple=powerpc-ibm-aix-xcoff -frame-pointer=all | FileCheck %s \
; RUN: -check-prefix=PPC32-AIX-FP

; RUN: llc -verify-machineinstrs < %s -mcpu=pwr4 -mattr=-altivec \
; RUN: -mtriple=powerpc64-ibm-aix-xcoff | FileCheck %s \
; RUN: -check-prefix=PPC64-AIX-NOFP

; RUN: llc -verify-machineinstrs < %s -mcpu=pwr4 -mattr=-altivec \
; RUN: -mtriple=powerpc64-ibm-aix-xcoff -frame-pointer=all | FileCheck %s \
; RUN: -check-prefix=PPC64-AIX-FP

define ptr @frame_small() {
        %tmp = alloca i32, i32 95
        ret ptr %tmp
}

; The linkage area, if there is one, is still on the top of the stack after
; `alloca` space.

; PPC32-LINUX-NOFP-LABEL: frame_small
; PPC32-LINUX-NOFP: stwu 1, -400(1)
; PPC32-LINUX-NOFP: addi 3, 1, 20
; PPC32-LINUX-NOFP: addi 1, 1, 400
; PPC32-LINUX-NOFP: blr

; PPC32-LINUX-FP-LABEL: frame_small
; PPC32-LINUX-FP: stwu 1, -400(1)
; PPC32-LINUX-FP: stw 31, 396(1)
; PPC32-LINUX-FP: mr 31, 1
; PPC32-LINUX-FP: addi 3, 31, 16
; PPC32-LINUX-FP: lwz 31, 396(1)
; PPC32-LINUX-FP: addi 1, 1, 400
; PPC32-LINUX-FP: blr

; PPC64-LINUX-NOFP-LABEL: frame_small
; PPC64-LINUX-NOFP: stdu 1, -432(1)
; PPC64-LINUX-NOFP: addi 3, 1, 52
; PPC64-LINUX-NOFP: addi 1, 1, 432
; PPC64-LINUX-NOFP: blr

; PPC64-LINUX-FP-LABEL: frame_small
; PPC64-LINUX-FP: std 31, -8(1)
; PPC64-LINUX-FP: stdu 1, -448(1)
; PPC64-LINUX-FP: mr 31, 1
; PPC64-LINUX-FP: addi 3, 31, 60
; PPC64-LINUX-FP: addi 1, 1, 448
; PPC64-LINUX-FP: ld 31, -8(1)
; PPC64-LINUX-FP: blr

; PPC32-AIX-NOFP-LABEL: frame_small
; PPC32-AIX-NOFP:      stwu 1, -416(1)
; PPC32-AIX-NOFP-NEXT: addi 3, 1, 36
; PPC32-AIX-NOFP-NEXT: addi 1, 1, 416
; PPC32-AIX-NOFP-NEXT: blr

; PPC32-AIX-FP-LABEL: frame_small
; PPC32-AIX-FP:      stw 31, -4(1)
; PPC32-AIX-FP-NEXT: stwu 1, -416(1)
; PPC32-AIX-FP-NEXT: mr 31, 1
; PPC32-AIX-FP-NEXT: addi 3, 31, 32
; PPC32-AIX-FP-NEXT: addi 1, 1, 416
; PPC32-AIX-FP-NEXT: lwz 31, -4(1)
; PPC32-AIX-FP-NEXT: blr

; PPC64-AIX-NOFP-LABEL: frame_small
; PPC64-AIX-NOFP:      stdu 1, -432(1)
; PPC64-AIX-NOFP-NEXT: addi 3, 1, 52
; PPC64-AIX-NOFP-NEXT: addi 1, 1, 432
; PPC64-AIX-NOFP-NEXT: blr

; PPC64-AIX-FP-LABEL: frame_small
; PPC64-AIX-FP:      std 31, -8(1)
; PPC64-AIX-FP-NEXT: stdu 1, -448(1)
; PPC64-AIX-FP-NEXT: mr 31, 1
; PPC64-AIX-FP-NEXT: addi 3, 31, 60
; PPC64-AIX-FP-NEXT: addi 1, 1, 448
; PPC64-AIX-FP-NEXT: ld 31, -8(1)
; PPC64-AIX-FP-NEXT: blr
