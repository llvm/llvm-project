; RUN: llc -mtriple=powerpc64-ibm-aix-xcoff %s --filetype=obj -o %t.o
; RUN: llvm-objdump -D -r --symbol-description %t.o | FileCheck %s --check-prefixes=CHECK,CHECK-NO-FS

; RUN: llc -mtriple=powerpc64-ibm-aix-xcoff %s --function-sections --filetype=obj -o %t.o
; RUN: llvm-objdump -D -r --symbol-description %t.o | FileCheck %s --check-prefixes=CHECK,CHECK-FS

; RUN: llc -mtriple=powerpc64-ibm-aix-xcoff --function-sections --code-model=large --filetype=obj %s -o %t.o
; RUN: llvm-objdump -D -r --symbol-description %t.o | FileCheck %s --check-prefixes=CHECK-LARGE,CHECK-LARGE-FS

; RUN: llc -mtriple=powerpc64-ibm-aix-xcoff --code-model=large --filetype=obj %s -o %t.o
; RUN: llvm-objdump -D -r --symbol-description %t.o | FileCheck %s --check-prefixes=CHECK-LARGE

; CHECK: Disassembly of section .text:
;;;; R_REF relocations associating .foo to (1) the __init_ifuncs constructor
;;;; and (2) the __update_foo variable.
; CHECK-NO-FS:          0000000000000000:  R_REF  {{.*}} __ifunc_sec[RW]
; CHECK-NO-FS-NEXT:     0000000000000000:  R_REF  {{.*}} .__init_ifuncs[PR]

;;;; .foo ifunc stub
; CHECK-NO-FS:   .foo:
; CHECK-NO-FS-NEXT:   ld 12, 8(2)
; CHECK-NO-FS-NEXT:      R_TOC {{.*}} foo[TC]
; CHECK-NO-FS-NEXT:   ld 11, 16(12)
; CHECK-NO-FS-NEXT:   ld 12, 0(12)
; CHECK-NO-FS-NEXT:   mtctr 12
; CHECK-NO-FS-NEXT:   bctr

; CHECK-FS:   .foo[PR]:
; CHECK-FS-NEXT:   ld 12, 8(2)
; CHECK-FS-NEXT:      R_REF {{.*}} __ifunc_sec[RW]
; CHECK-FS-NEXT:      R_REF {{.*}} .__init_ifuncs[PR]
; CHECK-FS-NEXT:      R_TOC {{.*}} foo[TC]
; CHECK-FS-NEXT:   ld 11, 16(12)
; CHECK-FS-NEXT:   ld 12, 0(12)
; CHECK-FS-NEXT:   mtctr 12
; CHECK-FS-NEXT:   bctr

; CHECK-LARGE: {{\.foo|\.foo\[PR\]}}:
; CHECK-LARGE-NEXT: addis 12, 2, 0
; CHECK-LARGE-FS-NEXT: R_REF {{.*}} __ifunc_sec[RW]
; CHECK-LARGE-FS-NEXT: R_REF {{.*}} .__init_ifuncs[PR]
; CHECK-LARGE-NEXT:    R_TOCU {{.*}} foo[TE]
; CHECK-LARGE-NEXT: ld 12, 8(12)
; CHECK-LARGE-NEXT:    R_TOCL {{.*}} foo[TE]
; CHECK-LARGE-NEXT: ld 11, 16(12)
; CHECK-LARGE-NEXT: ld 12, 0(12)
; CHECK-LARGE-NEXT: mtctr 12
; CHECK-LARGE-NEXT: bctr

; CHECK:   Disassembly of section .data:
;;;; section __ifunc_sec holding the [foo:foo_resolver] pairs
;;;; @__update_foo = private global { ptr, ptr } { ptr @foo, ptr @foo.resolver }, section "__ifunc_sec", align 8
; CHECK:  {{.*}} __ifunc_sec[RW]:
; CHECK-NEXT:  00 00 00 00   <unknown>
; CHECK-NEXT:       R_POS  {{.*}} foo[DS]
; CHECK-NEXT:  {{.*}}  <unknown>
; CHECK-NEXT:  00 00 00 00   <unknown>
; CHECK-NEXT:       R_POS  {{.*}} foo.resolver[DS]

;;;; A function descriptor for foo
; CHECK:  {{.*}} foo[DS]:
; CHECK-NEXT:  00 00 00 00   <unknown>
; CHECK-NEXT:       R_POS  {{.*}} .foo
; CHECK-NEXT:  {{.*}}  <unknown>
; CHECK-NEXT:  00 00 00 00  <unknown>
; CHECK-NEXT:       R_POS  {{.*}} TOC[TC0]

;;;; foo's TOC
; CHECK: {{.*}} foo[TC]:
; CHECK-NEXT:  00 00 00 00   <unknown>
; CHECK-NEXT:       R_POS  {{.*}} foo[DS]
; CHECK-NEXT:  {{.*}}  <unknown>

; CHECK-LARGE: {{.*}} foo[TE]:
; CHECK-LARGE-NEXT:  <unknown>
; CHECK-LARGE-NEXT:    R_POS {{.*}} foo[DS]
; CHECK-LARGE-NEXT:  <unknown>

@foo = ifunc i32 (...), ptr @foo.resolver

define hidden i32 @my_foo() {
entry:
  ret i32 4
}

define internal ptr @foo.resolver() {
entry:
  ret ptr @my_foo
}
