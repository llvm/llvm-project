; RUN: llc -mtriple=powerpc64-ibm-aix-xcoff %s -o - | FileCheck %s --check-prefixes=COMMON,NO-FUNCSECT -DALIGN=3 -DPTR_SIZE=8 -DLOAD=ld -DOFF=16
; RUN: llc -mtriple=powerpc-ibm-aix-xcoff %s -o - | FileCheck %s --check-prefixes=COMMON,NO-FUNCSECT -DALIGN=2 -DPTR_SIZE=4 -DLOAD=lwz -DOFF=8

; RUN: llc -mtriple=powerpc64-ibm-aix-xcoff --function-sections %s -o - | FileCheck %s --check-prefixes=COMMON,FUNCSECT -DALIGN=3 -DPTR_SIZE=8 -DLOAD=ld -DOFF=16
; RUN: llc -mtriple=powerpc-ibm-aix-xcoff --function-sections %s -o - | FileCheck %s --check-prefixes=COMMON,FUNCSECT -DALIGN=2 -DPTR_SIZE=4 -DLOAD=lwz -DOFF=8

; RUN: llc -mtriple=powerpc64-ibm-aix-xcoff --function-sections --code-model=large %s -o - | FileCheck %s --check-prefixes=LARGE -DALIGN=3 -DPTR_SIZE=8 -DLOAD=ld -DOFF=16
; RUN: llc -mtriple=powerpc-ibm-aix-xcoff --function-sections --code-model=large %s -o - | FileCheck %s --check-prefixes=LARGE -DALIGN=2 -DPTR_SIZE=4 -DLOAD=lwz -DOFF=8

;;;; section __ifunc_sec holding the [foo:foo_resolver] pairs
; COMMON:              .csect __ifunc_sec[RW],2
; COMMON-NEXT:         .align  [[ALIGN]]
; COMMON-NEXT: L..__update_foo:
; COMMON-NEXT:         .vbyte  [[PTR_SIZE]], foo[DS]
; COMMON-NEXT:         .vbyte  [[PTR_SIZE]], foo.resolver[DS]

;;;; forward declare the __init_ifuncs constructor
; COMMON-NEXT:         .extern .__init_ifuncs[PR]

;;;; declare foo[DS] and .foo[PR]
; FUNCSECT-NEXT:       .csect .foo[PR],5
; FUNCSECT-NEXT:       .ref L..__update_foo
; FUNCSECT-NEXT:       .ref .__init_ifuncs[PR]
; FUNCSECT-NEXT:       .globl  foo[DS]
; FUNCSECT-NEXT:       .globl  .foo[PR]
; FUNCSECT-NEXT:       .align  2

; NO-FUNCSECT-NEXT:    .csect ..text..[PR],5
; NO-FUNCSECT-NEXT:    .ref L..__update_foo
; NO-FUNCSECT-NEXT:    .ref .__init_ifuncs[PR]
; NO-FUNCSECT-NEXT:    .globl  foo[DS]
; NO-FUNCSECT-NEXT:    .globl  .foo
; NO-FUNCSECT-NEXT:    .align  2

;;;; define foo's descriptor
; COMMON-NEXT:         .csect foo[DS],[[ALIGN]]
; FUNCSECT-NEXT:       .vbyte  [[PTR_SIZE]], .foo[PR]
; NO-FUNCSECT-NEXT:    .vbyte  [[PTR_SIZE]], .foo
; COMMON-NEXT:         .vbyte  [[PTR_SIZE]], TOC[TC0]
; COMMON-NEXT:         .vbyte  [[PTR_SIZE]], 0

;;;; emit foo's body
; FUNCSECT-NEXT:       .csect .foo[PR],5
; NO-FUNCSECT-NEXT:    .csect ..text..[PR],5
; NO-FUNCSECT-NEXT: .foo:
; COMMON-NEXT:         [[LOAD]] 12, [[FOO_TOC:.*]](2)
; COMMON-NEXT:         [[LOAD]] 11, [[OFF]](12)
; COMMON-NEXT:         [[LOAD]] 12, 0(12)
; COMMON-NEXT:         mtctr 12
; COMMON-NEXT:         bctr

; -mcmodel=large:
; LARGE:             .csect .foo[PR],5
; LARGE:             addis 12, [[FOO_TOC:.*]]@u(2)
; LARGE-NEXT:        [[LOAD]] 12, [[FOO_TOC]]@l(12)
; LARGE-NEXT:        [[LOAD]] 11, [[OFF]](12)
; LARGE-NEXT:        [[LOAD]] 12, 0(12)

;;;; foo's TOC entry
; COMMON: [[FOO_TOC]]:
; COMMON-NEXT:         .tc foo[TC],foo[DS]
; LARGE: [[FOO_TOC]]:
; LARGE-NEXT:          .tc foo[TE],foo[DS]

@foo = ifunc i32 (...), ptr @foo.resolver

define hidden i32 @my_foo() {
entry:
  ret i32 4
}

define internal ptr @foo.resolver() {
entry:
  ret ptr @my_foo
}
