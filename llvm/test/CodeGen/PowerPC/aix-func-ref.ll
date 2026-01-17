; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff < %s | \
; RUN: FileCheck %s -check-prefixes=NOFSECTS,CHECK

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff --function-sections < %s | \
; RUN: FileCheck %s -check-prefixes=FSECTS,CHECK

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -filetype=obj -o %t.o < %s
; RUN: llvm-objdump -D -r --symbol-description %t.o | FileCheck -check-prefix=OBJ %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff --function-sections -filetype=obj -o %t.o < %s
; RUN: llvm-objdump -D -r --symbol-description %t.o | FileCheck -check-prefix=FSECTOBJ %s

@a = global i32 1
@b = global i32 2
@c = global i32 3

define i32 @foo() !implicit.ref !0 {
  ret i32 0
}

define i32 @bar() !implicit.ref !1 !implicit.ref !2 {
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

; OBJ: Disassembly of section .text:
; OBJ: .foo:
; OBJ:   li 3, 0
; OBJ:   R_REF {{.*}} a[RW]
; OBJ:   R_REF {{.*}} b[RW]
; OBJ:   R_REF {{.*}} c[RW]
; OBJ:   blr
; OBJ: .bar

; FSECTOBJ: .foo[PR]:
; FSECTOBJ:   li 3, 0
; FSECTOBJ:   R_REF {{.*}} a[RW]
; FSECTOBJ:   blr
; FSECTOBJ: .bar[PR]:
; FSECTOBJ:   li 3, 0
; FSECTOBJ:   R_REF {{.*}} b[RW]
; FSECTOBJ:   R_REF {{.*}} c[RW]
; FSECTOBJ:   blr
