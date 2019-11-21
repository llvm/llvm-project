; This file tests the codegen of mergeable strings in AIX assembly only.
; Once the codegen of mergeable strings for XCOFF object files is supported
; the test in this file should be merged into aix-xcoff-data.ll with additional
; tests for XCOFF object files.

; RUN: llc -verify-machineinstrs -mcpu=pwr7 \
; RUN:     -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 \
; RUN:     -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s

@strA = private unnamed_addr constant [14 x i8] c"hello world!\0A\00", align 1

; CHECK:      .csect .rodata.str1.1[RO]
; CHECK-NEXT: .LstrA:
; CHECK-NEXT: .byte   104
; CHECK-NEXT: .byte   101
; CHECK-NEXT: .byte   108
; CHECK-NEXT: .byte   108
; CHECK-NEXT: .byte   111
; CHECK-NEXT: .byte   32
; CHECK-NEXT: .byte   119
; CHECK-NEXT: .byte   111
; CHECK-NEXT: .byte   114
; CHECK-NEXT: .byte   108
; CHECK-NEXT: .byte   100
; CHECK-NEXT: .byte   33
; CHECK-NEXT: .byte   10
; CHECK-NEXT: .byte   0
