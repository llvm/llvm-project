; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s

@ivar = local_unnamed_addr global i32 35, align 4
@llvar = local_unnamed_addr global i64 36, align 8
@svar = local_unnamed_addr global i16 37, align 2
@fvar = local_unnamed_addr global float 8.000000e+02, align 4
@dvar = local_unnamed_addr global double 9.000000e+02, align 8
@over_aligned = local_unnamed_addr global double 9.000000e+02, align 32
@charr = local_unnamed_addr global [4 x i8] c"abcd", align 1
@dblarr = local_unnamed_addr global [4 x double] [double 1.000000e+00, double 2.000000e+00, double 3.000000e+00, double 4.000000e+00], align 8

; CHECK:      .csect .data[RW]
; CHECK-NEXT: .globl  ivar
; CHECK-NEXT: .align  2
; CHECK-NEXT: ivar:
; CHECK-NEXT: .long   35

; CHECK:      .globl  llvar
; CHECK-NEXT: .align  3
; CHECK-NEXT: llvar:
; CHECK-NEXT: .llong  36

; CHECK:      .globl  svar
; CHECK-NEXT: .align  1
; CHECK-NEXT: svar:
; CHECK-NEXT: .short  37

; CHECK:      .globl  fvar
; CHECK-NEXT: .align  2
; CHECK-NEXT: fvar:
; CHECK-NEXT: .long   1145569280

; CHECK:      .globl  dvar
; CHECK-NEXT: .align  3
; CHECK-NEXT: dvar:
; CHECK-NEXT: .llong  4651127699538968576

; CHECK:      .globl  over_aligned
; CHECK-NEXT: .align  5
; CHECK-NEXT: over_aligned:
; CHECK-NEXT: .llong  4651127699538968576

; CHECK:      .globl  charr
; CHECK-NEXT: charr:
; CHECK-NEXT: .byte   97
; CHECK-NEXT: .byte   98
; CHECK-NEXT: .byte   99
; CHECK-NEXT: .byte   100

; CHECK:      .globl  dblarr
; CHECK-NEXT: .align  3
; CHECK-NEXT: dblarr:
; CHECK-NEXT: .llong  4607182418800017408
; CHECK-NEXT: .llong  4611686018427387904
; CHECK-NEXT: .llong  4613937818241073152
; CHECK-NEXT: .llong  4616189618054758400
