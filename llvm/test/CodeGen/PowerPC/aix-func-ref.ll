; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff < %s | \
; RUN: FileCheck %s -check-prefixes=NOFSECTS,CHECK

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff --function-sections < %s | \
; RUN: FileCheck %s -check-prefixes=FSECTS,CHECK

@a = global i32 1
@b = global i32 2
@c = global i32 3

define i32 @foo() !ref !0 {
  ret i32 0
}

define i32 @bar() !ref !1 !ref !2 {
  ret i32 0
}

!0 = !{ptr @a}
!1 = !{ptr @b}
!2 = !{ptr @c}

; NOFSECTS:  .foo:
; FSECTS:    .csect .foo[PR]
; CHECK:       .ref a[RW]

; NOFSECTS:  .bar:
; FSECTS:    .csect .bar[PR]
; CHECK:       .ref b[RW]
; CHECK:       .ref c[RW]

