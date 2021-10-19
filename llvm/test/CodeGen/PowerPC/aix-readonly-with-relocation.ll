; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc-ibm-aix-xcoff \
; RUN:     --relocation-model=pic -data-sections=false < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     --relocation-model=pic -data-sections=false < %s | FileCheck --check-prefix=CHECK64 %s

@a = common global i32 0
@b = constant i32* @a

;CHECK:         .comm   a[RW],4,2
;CHECK-NEXT:    .csect .data[RW],2
;CHECK-NEXT:    .globl  b
;CHECK-NEXT:    .align  2
;CHECK-NEXT: b:
;CHECK-NEXT:    .vbyte	4, a

;CHECK64:       .comm   a[RW],4,2
;CHECK64-NEXT:  .csect .data[RW],3
;CHECK64-NEXT:  .globl  b
;CHECK64-NEXT:  .align  3
;CHECK64-NEXT: b:
;CHECK64-NEXT:  .vbyte	8, a
