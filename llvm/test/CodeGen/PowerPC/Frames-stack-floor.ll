; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu | \
; RUN: FileCheck %s -check-prefix=PPC32-LINUX-NOFP

; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu \
; RUN: -frame-pointer=all | FileCheck %s -check-prefix=PPC32-LINUX-FP

; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu | \
; RUN: FileCheck %s -check-prefix=PPC64-NOFP

; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu \
; RUN: -frame-pointer=all | FileCheck %s -check-prefix=PPC64-FP

; RUN: llc -verify-machineinstrs < %s -mcpu=pwr4 -mattr=-altivec \
; RUN: -mtriple=powerpc-ibm-aix-xcoff | FileCheck %s -check-prefix=PPC32-AIX-NOFP

; RUN: llc -verify-machineinstrs < %s -mcpu=pwr4 -mattr=-altivec \
; RUN: -mtriple=powerpc-ibm-aix-xcoff -frame-pointer=all | FileCheck %s \
; RUN: -check-prefix=PPC32-AIX-FP

; RUN: llc -verify-machineinstrs < %s -mcpu=pwr4 -mattr=-altivec \
; RUN: -mtriple=powerpc64-ibm-aix-xcoff | FileCheck %s -check-prefix=PPC64-NOFP

; RUN: llc -verify-machineinstrs < %s -mcpu=pwr4 -mattr=-altivec \
; RUN: -mtriple=powerpc64-ibm-aix-xcoff -frame-pointer=all | FileCheck %s \
; RUN: -check-prefix=PPC64-FP

; - PPC64 SVR4ABI and AIX ABI:
;   288 bytes = 18*8 (FPRs) + 18*8 (GPRs, GPR13 reserved);
; - PPC32 SVR4ABI has no red zone;
; - PPC32 AIX ABI:
;   220 bytes = 18*8 (FPRs) + 19*4 (GPRs);

define ptr @in_stack_floor_32() {
        %tmp = alloca i32, i32 55
        ret ptr %tmp
}

define ptr @out_stack_floor_32() {
        %tmp = alloca i32, i32 56
        ret ptr %tmp
}

define ptr @in_stack_floor_64() {
        %tmp = alloca i32, i32 72
        ret ptr %tmp
}

define ptr @out_stack_floor_64() {
        %tmp = alloca i32, i32 73
        ret ptr %tmp
}

; PPC32-LINUX-NOFP-LABEL: in_stack_floor_32
; PPC32-LINUX-NOFP: stwu

; PPC32-LINUX-NOFP-LABEL: out_stack_floor_32
; PPC32-LINUX-NOFP: stwu

; PPC32-LINUX-FP-LABEL: in_stack_floor_32
; PPC32-LINUX-FP: stwu

; PPC32-LINUX-FP-LABEL: out_stack_floor_32
; PPC32-LINUX-FP: stwu

; PPC32-AIX-NOFP-LABEL: in_stack_floor_32
; PPC32-AIX-NOFP-NOT: stwu

; PPC32-AIX-NOFP-LABEL: out_stack_floor_32
; PPC32-AIX-NOFP:      stwu 1, -256(1)

; PPC32-AIX-FP-LABEL: in_stack_floor_32
; PPC32-AIX-FP: stwu 1, -256(1)

; PPC32-AIX-FP-LABEL: out_stack_floor_32
; PPC32-AIX-FP: stwu 1, -256(1)

; PPC64-NOFP-LABEL: in_stack_floor_64
; PPC64-NOFP:      addi 3, 1, -288

; PPC64-NOFP-LABEL: out_stack_floor_64
; PPC64-NOFP:      stdu 1, -352(1)

; PPC64-FP-LABEL: in_stack_floor_64
; PPC64-FP: stdu 1, -352(1)

; PPC64-FP-LABEL: out_stack_floor_64
; PPC64-FP: stdu 1, -352(1)
