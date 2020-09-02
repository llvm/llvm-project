; RUN: llc -verify-machineinstrs < %s -mcpu=pwr4 -mattr=-altivec \
; RUN:     -mtriple=powerpc-ibm-aix-xcoff | \
; RUN:   FileCheck %s -check-prefix=AIX32

; RUN: llc -verify-machineinstrs < %s -mcpu=pwr4 -mattr=-altivec \
; RUN:     -mtriple=powerpc64-ibm-aix-xcoff | \
; RUN:   FileCheck %s -check-prefixes=AIX64

declare void @clobber(i32*)

define dso_local float @frameptr_only(i32 %n, float %f) {
entry:
  %0 = alloca i32, i32 %n
  call void @clobber(i32* %0)
  ret float %f
}

; AIX32: stw 31, -12(1)
; AIX32: stwu 1, -80(1)
; AIX32: lwz 1, 0(1)
; AIX32: lwz 31, -12(1)

; AIX64: std 31, -16(1)
; AIX64: stdu 1, -144(1)
; AIX64: ld 1, 0(1)
; AIX64: ld 31, -16(1)

