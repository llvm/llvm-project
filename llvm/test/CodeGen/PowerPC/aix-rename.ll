; RUN: llc --function-sections -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck --check-prefix=NOFUNCSECT %s

@a = global i32 1, section "abcd", !rename !0
@b = global i32 2, section "abcd", !rename !0
@c = global i32 3, section "abcd", !rename !0
@d = global i32 4, section "abcd", !rename !0

!0 = !{}

;CHECK:     .csect abcd.a[RW]
;CHECK:     .globl  a

;CHECK:     .csect abcd.b[RW]
;CHECK:     .globl  b

;CHECK:     .csect abcd.c[RW]
;CHECK:     .globl  c

;CHECK:     .csect abcd.d[RW]
;CHECK:     .globl  d

;CHECK:     .rename abcd.a[RW],"abcd"
;CHECK:     .rename abcd.b[RW],"abcd"
;CHECK:     .rename abcd.c[RW],"abcd"
;CHECK:     .rename abcd.d[RW],"abcd"

;NOFUNCSECT:     .csect abcd[RW],2
;NOFUNCSECT-NOT: .csect
;NOFUNCSECT:     .globl  a
;NOFUNCSECT-NOT: .csect
;NOFUNCSECT:     .globl  b
;NOFUNCSECT-NOT: .csect
;NOFUNCSECT:     .globl  c
;NOFUNCSECT-NOT: .csect
;NOFUNCSECT:     .globl  d

;NOFUNCSECT-NOT: .rename
