; RUN: llc -verify-machineinstrs -O0 -mcpu=pwr4 -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mcpu=pwr4 -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s

@a = constant [4 x i8] c"\02\02\02\02", align 1

; CHECK-NOT:  .space  4,2
; CHECK:      .byte   2
; CHECK-NEXT: .byte   2
; CHECK-NEXT: .byte   2
; CHECK-NEXT: .byte   2
