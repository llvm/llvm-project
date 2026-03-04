; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff \
; RUN:   -data-sections=false -function-sections=false -filetype=obj -o %t.o < %s
; RUN: llvm-objdump -D -r --symbol-description %t.o | FileCheck %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff \
; RUN:   -data-sections=false -function-sections=false -filetype=obj -o %t.o < %s
; RUN: llvm-objdump -D -r --symbol-description %t.o | FileCheck %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff \
; RUN:   -data-sections=true -function-sections=true -filetype=obj -o %t.o < %s
; RUN: llvm-objdump -D -r --symbol-description %t.o | FileCheck %s --check-prefix=SEC

@a = global i32 1
@b = dso_local constant i32 2
@p = dso_local global ptr @a, !implicit.ref !0
@c = global i32 3

define dso_local void @foo(i32 noundef signext %i) {
entry:
  tail call void @extern_func1(i32 noundef signext %i)
  ret void
}

declare void @extern_func1(i32 noundef signext)

define dso_local signext i32 @bar() {
entry:
  %0 = load i32, ptr @a, align 4
  %call = tail call signext i32 @extern_func2(i32 noundef signext %0)
  ret i32 %call
}

declare signext i32 @extern_func2(i32 noundef signext)

define dso_local signext i32 @baz() !implicit.ref !0 {
entry:
  %0 = load i32, ptr @c, align 4
  ret i32 %0
}

!0 = !{ptr @b}

; CHECK: Disassembly of section .text:
; CHECK-EMPTY:
; CHECK-NEXT: 00000000 (idx: {{[0-9]+}}) .foo:
; CHECK:      00000000:  R_REF	(idx: {{[0-9]+}}) b
; CHECK:      0000000c:  R_RBR	(idx: {{[0-9]+}}) .extern_func1[PR]
; CHECK:      0000004a:  R_TOC	(idx: {{[0-9]+}}) a[TC]
; CHECK:      00000054:  R_RBR	(idx: {{[0-9]+}}) .extern_func2[PR]
; CHECK:      00000092:  R_TOC   (idx: {{[0-9]+}}) c[TC]

; CHECK: Disassembly of section .data:
; CHECK-EMPTY:
; CHECK-NEXT: [[DATA:[0-9a-f]+]] (idx: {{[0-9]+}}) a:
; CHECK:      [[DATA]]:  R_REF	(idx: {{[0-9]+}}) b
; CHECK:      [[DATA_P:[0-9a-f]+]] (idx: {{[0-9]+}}) p:
; CHECK:      [[DATA_P]]:  R_POS	(idx: {{[0-9]+}}) a
; CHECK:      {{[0-9a-f]+}} (idx: {{[0-9]+}}) c:

; SEC: Disassembly of section .text
; SEC-EMPTY:
; SEC-NEXT: .foo[PR]:
; SEC:      R_RBR        (idx: {{[0-9]+}}) .extern_func1[PR]
; SEC:      .bar[PR]:
; SEC:      R_TOC        (idx: {{[0-9]+}}) a[TC]
; SEC:      R_RBR        (idx: {{[0-9]+}}) .extern_func2[PR]
; SEC:      .baz[PR]:
; SEC:      R_REF        (idx: {{[0-9]+}}) b[RO]
; SEC:      R_TOC        (idx: {{[0-9]+}}) c[TC]

; SEC: Disassembly of section .data:
; SEC-EMPTY:
; SEC-NEXT: a[RW]:
; SEC:      p[RW]:
; SEC:      R_REF        (idx: {{[0-9]+}}) b[RO]
; SEC:      R_POS        (idx: {{[0-9]+}}) a[RW]
; SEC:      c[RW]:
