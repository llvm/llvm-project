; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck --check-prefix=CHECK64 %s

@foo_ptr = global void (...)* @foo
declare void @foo(...)

@bar_ptr1 = global void (...)* bitcast (void ()* @bar to void (...)*)
define void @bar() {
entry:
  ret void
}


;CHECK:          .csect .data[RW],2
;CHECK-NEXT:     .globl  foo_ptr
;CHECK-NEXT:     .align  2
;CHECK-NEXT:     foo_ptr:
;CHECK-NEXT:     .vbyte	4, foo[DS]
;CHECK-NEXT:     .globl  bar_ptr1
;CHECK-NEXT:     .align  2
;CHECK-NEXT:     bar_ptr1:
;CHECK-NEXT:     .vbyte	4, bar[DS]
;CHECK-NEXT:     .extern foo[DS]

;CHECK64:         .csect .data[RW],3
;CHECK64-NEXT:         .globl  foo_ptr
;CHECK64-NEXT:         .align  3
;CHECK64-NEXT:    foo_ptr:
;CHECK64-NEXT:         .vbyte	8, foo[DS]
;CHECK64-NEXT:         .globl  bar_ptr1
;CHECK64-NEXT:         .align  3
;CHECK64-NEXT:    bar_ptr1:
;CHECK64-NEXT:         .vbyte	8, bar[DS]
;CHECK64-NEXT:         .extern foo[DS]
