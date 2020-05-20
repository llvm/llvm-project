; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s -check-prefix=PPC32-LINUX
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s -check-prefix=PPC64-LINUX
; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu -frame-pointer=all | FileCheck %s -check-prefix=PPC32-LINUX
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -frame-pointer=all | FileCheck %s -check-prefix=PPC64-LINUX
; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s -check-prefix=PPC32-LINUX

; RUN: llc < %s -mcpu=pwr4 -mattr=-altivec -verify-machineinstrs \
; RUN: -mtriple=powerpc-ibm-aix-xcoff | FileCheck %s -check-prefix=PPC32-AIX

; RUN: llc < %s -mcpu=pwr4 -mattr=-altivec -verify-machineinstrs \
; RUN: -mtriple=powerpc-ibm-aix-xcoff -frame-pointer=all | FileCheck %s \
; RUN: -check-prefix=PPC32-AIX

; RUN: llc < %s -mcpu=pwr4 -mattr=-altivec -verify-machineinstrs \
; RUN: -mtriple=powerpc64-ibm-aix-xcoff | FileCheck %s -check-prefix=PPC64-AIX

; RUN: llc < %s -mcpu=pwr4 -mattr=-altivec -verify-machineinstrs \
; RUN: -mtriple=powerpc64-ibm-aix-xcoff -frame-pointer=all | FileCheck %s \
; RUN: -check-prefix=PPC64-AIX

define i32* @f1(i32 %n) nounwind {
        %tmp = alloca i32, i32 %n               ; <i32*> [#uses=1]
        ret i32* %tmp
}

; PPC32-LINUX-LABEL: f1
; PPC32-LINUX:      stwu 1, -32(1)
; PPC32-LINUX-NEXT: slwi 3, 3, 2
; PPC32-LINUX-NEXT: addi 3, 3, 15
; PPC32-LINUX-NEXT: stw 31, 28(1)
; PPC32-LINUX-NEXT: mr 31, 1
; PPC32-LINUX-NEXT: rlwinm 3, 3, 0, 0, 27
; PPC32-LINUX-NEXT: neg 3, 3
; PPC32-LINUX-NEXT: addi 4, 31, 32
; PPC32-LINUX-NEXT: stwux 4, 1, 3
; PPC32-LINUX-NEXT: lwz 31, 0(1)
; PPC32-LINUX-NEXT: addi 3, 1, 16
; PPC32-LINUX-NEXT: lwz 0, -4(31)
; PPC32-LINUX-NEXT: mr 1, 31
; PPC32-LINUX-NEXT: mr 31, 0
; PPC32-LINUX-NEXT: blr

; PPC64-LINUX-LABEL: f1
; PPC64-LINUX:      std 31, -8(1)
; PPC64-LINUX-NEXT: stdu 1, -64(1)
; PPC64-LINUX-NEXT: rldic 3, 3, 2, 30
; PPC64-LINUX-NEXT: mr 31, 1
; PPC64-LINUX-NEXT: addi 3, 3, 15
; PPC64-LINUX-NEXT: rldicl 3, 3, 60, 4
; PPC64-LINUX-NEXT: addi 4, 31, 64
; PPC64-LINUX-NEXT: rldicl 3, 3, 4, 29
; PPC64-LINUX-NEXT: neg 3, 3
; PPC64-LINUX-NEXT: stdux 4, 1, 3

; The linkage area is always put on the top of the stack.
; PPC64-LINUX-NEXT: addi 3, 1, 48

; PPC64-LINUX-NEXT: ld 1, 0(1)
; PPC64-LINUX-NEXT: ld 31, -8(1)
; PPC64-LINUX-NEXT: blr

; PPC32-AIX-LABEL: f1
; PPC32-AIX:      stw 31, -4(1)
; PPC32-AIX-NEXT: stwu 1, -48(1)
; PPC32-AIX-NEXT: slwi 3, 3, 2
; PPC32-AIX-NEXT: mr 31, 1
; PPC32-AIX-NEXT: addi 3, 3, 15
; PPC32-AIX-NEXT: addi 4, 31, 48
; PPC32-AIX-NEXT: rlwinm 3, 3, 0, 0, 27
; PPC32-AIX-NEXT: neg 3, 3
; PPC32-AIX-NEXT: stwux 4, 1, 3

; The linkage area is always put on the top of the stack.
; PPC32-AIX-NEXT: addi 3, 1, 32

; PPC32-AIX-NEXT: lwz 1, 0(1)
; PPC32-AIX-NEXT: lwz 31, -4(1)
; PPC32-AIX-NEXT: blr

; PPC64-AIX-LABEL: f1
; PPC64-AIX:      std 31, -8(1)
; PPC64-AIX-NEXT: stdu 1, -64(1)
; PPC64-AIX-NEXT: rldic 3, 3, 2, 30
; PPC64-AIX-NEXT: mr 31, 1
; PPC64-AIX-NEXT: addi 3, 3, 15
; PPC64-AIX-NEXT: addi 4, 31, 64
; PPC64-AIX-NEXT: rldicl 3, 3, 60, 4 
; PPC64-AIX-NEXT: rldicl 3, 3, 4, 29
; PPC64-AIX-NEXT: neg 3, 3
; PPC64-AIX-NEXT: stdux 4, 1, 3

; The linkage area is always put on the top of the stack.
; PPC64-AIX-NEXT: addi 3, 1, 48

; PPC64-AIX-NEXT: ld 1, 0(1)
; PPC64-AIX-NEXT: ld 31, -8(1)
; PPC64-AIX-NEXT: blr
