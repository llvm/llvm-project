; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc-ibm-aix-xcoff  < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s

@const_ivar = constant i32 35, align 4
@const_llvar = constant i64 36, align 8
@const_svar = constant i16 37, align 2
@const_fvar = constant float 8.000000e+02, align 4
@const_dvar = constant double 9.000000e+02, align 8
@const_over_aligned = constant double 9.000000e+02, align 32
@const_chrarray = constant [4 x i8] c"abcd", align 1
@const_dblarr = constant [4 x double] [double 1.000000e+00, double 2.000000e+00, double 3.000000e+00, double 4.000000e+00], align 8

; CHECK:               .csect .rodata[RO]
; CHECK-NEXT:          .globl  const_ivar
; CHECK-NEXT:          .align  2
; CHECK-NEXT:  const_ivar:
; CHECK-NEXT:          .long   35
; CHECK-NEXT:          .globl  const_llvar
; CHECK-NEXT:          .align  3
; CHECK-NEXT:  const_llvar:
; CHECK-NEXT:          .llong  36
; CHECK-NEXT:          .globl  const_svar
; CHECK-NEXT:          .align  1
; CHECK-NEXT:  const_svar:
; CHECK-NEXT:          .short  37
; CHECK-NEXT:          .globl  const_fvar
; CHECK-NEXT:          .align  2
; CHECK-NEXT:  const_fvar:
; CHECK-NEXT:          .long   1145569280
; CHECK-NEXT:          .globl  const_dvar
; CHECK-NEXT:          .align  3
; CHECK-NEXT:  const_dvar:
; CHECK-NEXT:          .llong  4651127699538968576
; CHECK-NEXT:          .globl  const_over_aligned
; CHECK-NEXT:          .align  5
; CHECK-NEXT:  const_over_aligned:
; CHECK-NEXT:          .llong  4651127699538968576
; CHECK-NEXT:          .globl  const_chrarray
; CHECK-NEXT:  const_chrarray:
; CHECK-NEXT:          .byte   97
; CHECK-NEXT:          .byte   98
; CHECK-NEXT:          .byte   99
; CHECK-NEXT:          .byte   100
; CHECK-NEXT:          .globl  const_dblarr
; CHECK-NEXT:          .align  3
; CHECK-NEXT:  const_dblarr:
; CHECK-NEXT:          .llong  4607182418800017408
; CHECK-NEXT:          .llong  4611686018427387904
; CHECK-NEXT:          .llong  4613937818241073152
; CHECK-NEXT:          .llong  4616189618054758400
