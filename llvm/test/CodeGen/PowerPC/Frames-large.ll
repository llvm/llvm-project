; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu | \
; RUN: FileCheck %s -check-prefix=PPC32-LINUX-NOFP

; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu \
; RUN: -frame-pointer=all | FileCheck %s -check-prefix=PPC32-LINUX-FP

; RUN: llc -verify-machineinstrs < %s -mcpu=ppc -mtriple=powerpc64-unknown-linux-gnu | \
; RUN: FileCheck %s -check-prefix=PPC64-LINUX-NOFP

; RUN: llc -verify-machineinstrs < %s -mcpu=ppc -mtriple=powerpc64-unknown-linux-gnu \
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

define ptr @f1() nounwind {
        %tmp = alloca i32, i32 8191             ; <ptr> [#uses=1]
        ret ptr %tmp
}

;   - The stdux is used to update the back-chain link when allocated frame is large
;     that we can not address it by a 16-bit signed integer;
;   - The linkage area, if there is one, is still on the top of the stack after
;     `alloca` space.

; PPC32-LINUX-NOFP-LABEL: f1
; PPC32-LINUX-NOFP:      lis 0, -1
; PPC32-LINUX-NOFP-NEXT: ori 0, 0, 32752
; PPC32-LINUX-NOFP-NEXT: stwux 1, 1, 0
; PPC32-LINUX-NOFP-NOT:  stwux
; PPC32-LINUX-NOFP:      mr 0, 31
; PPC32-LINUX-NOFP-DAG:  addi 3, 1, 20
; PPC32-LINUX-NOFP-DAG:  lwz 31, 0(1)
; PPC32-LINUX-NOFP-NEXT: mr 1, 31
; PPC32-LINUX-NOFP-NEXT: mr 31, 0
; PPC32-LINUX-NOFP-NEXT: blr

; PPC32-LINUX-FP-LABEL: f1
; PPC32-LINUX-FP:      lis 0, -1
; PPC32-LINUX-FP-NEXT: ori 0, 0, 32752
; PPC32-LINUX-FP-NEXT: stwux 1, 1, 0
; PPC32-LINUX-FP-NOT:  stwux
; PPC32-LINUX-FP:      mr 31, 1
; PPC32-LINUX-FP-NEXT: addi 3, 31, 16
; PPC32-LINUX-FP-NEXT: lwz 31, 0(1)
; PPC32-LINUX-FP-NEXT: lwz 0, -4(31)
; PPC32-LINUX-FP-NEXT: mr 1, 31
; PPC32-LINUX-FP-NEXT: mr 31, 0
; PPC32-LINUX-FP-NEXT: blr

; PPC64-LINUX-NOFP-LABEL: f1:
; PPC64-LINUX-NOFP:      lis 0, -1
; PPC64-LINUX-NOFP-NEXT: ori 0, 0, 32720
; PPC64-LINUX-NOFP-NEXT: stdux 1, 1, 0
; PPC64-LINUX-NOFP-NEXT: addi 3, 1, 52
; PPC64-LINUX-NOFP-NEXT: ld 1, 0(1)
; PPC64-LINUX-NOFP-NEXT: blr

; PPC64-LINUX-FP-LABEL: f1:
; PPC64-LINUX-FP:      lis 0, -1
; PPC64-LINUX-FP-NEXT: ori 0, 0, 32704
; PPC64-LINUX-FP-NEXT: std 31, -8(1)
; PPC64-LINUX-FP-NEXT: stdux 1, 1, 0
; PPC64-LINUX-FP-NEXT: mr 31, 1
; PPC64-LINUX-FP-NEXT: addi 3, 31, 60
; PPC64-LINUX-FP-NEXT: ld 1, 0(1)
; PPC64-LINUX-FP-NEXT: ld 31, -8(1)
; PPC64-LINUX-FP-NEXT: blr

; PPC32-AIX-NOFP-LABEL: f1
; PPC32-AIX-NOFP:      lis 0, -1
; PPC32-AIX-NOFP-NEXT: ori 0, 0, 32736
; PPC32-AIX-NOFP-NEXT: stwux 1, 1, 0
; PPC32-AIX-NOFP-NEXT: addi 3, 1, 36
; PPC32-AIX-NOFP-NEXT: lwz 1, 0(1)
; PPC32-AIX-NOFP-NEXT: blr

; PPC32-AIX-FP-LABEL: f1
; PPC32-AIX-FP:      lis 0, -1
; PPC32-AIX-FP-NEXT: stw 31, -4(1)
; PPC32-AIX-FP-NEXT: ori 0, 0, 32736
; PPC32-AIX-FP-NEXT: stwux 1, 1, 0
; PPC32-AIX-FP-NEXT: mr 31, 1
; PPC32-AIX-FP-NEXT: addi 3, 31, 32
; PPC32-AIX-FP-NEXT: lwz 1, 0(1)
; PPC32-AIX-FP-NEXT: lwz 31, -4(1)
; PPC32-AIX-FP-NEXT: blr

; PPC64-AIX-NOFP-LABEL: f1
; PPC64-AIX-NOFP:      lis 0, -1
; PPC64-AIX-NOFP-NEXT: ori 0, 0, 32720
; PPC64-AIX-NOFP-NEXT: stdux 1, 1, 0
; PPC64-AIX-NOFP-NEXT: addi 3, 1, 52
; PPC64-AIX-NOFP-NEXT: ld 1, 0(1)
; PPC64-AIX-NOFP-NEXT: blr

; PPC64-AIX-FP-LABEL: f1
; PPC64-AIX-FP:      lis 0, -1
; PPC64-AIX-FP-NEXT: std 31, -8(1)
; PPC64-AIX-FP-NEXT: ori 0, 0, 32704
; PPC64-AIX-FP-NEXT: stdux 1, 1, 0
; PPC64-AIX-FP-NEXT: mr 31, 1
; PPC64-AIX-FP-NEXT: addi 3, 31, 60
; PPC64-AIX-FP-NEXT: ld 1, 0(1)
; PPC64-AIX-FP-NEXT: ld 31, -8(1)
; PPC64-AIX-FP-NEXT: blr
