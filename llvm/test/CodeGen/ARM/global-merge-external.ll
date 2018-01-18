; RUN: llc < %s -mtriple=arm-eabi  -arm-global-merge                                 | FileCheck %s --check-prefixes=CHECK,CHECK-MERGE
; RUN: llc < %s -mtriple=arm-eabi  -arm-global-merge -global-merge-on-external=true  | FileCheck %s --check-prefixes=CHECK,CHECK-MERGE
; RUN: llc < %s -mtriple=arm-eabi  -arm-global-merge -global-merge-on-external=false | FileCheck %s --check-prefixes=CHECK,CHECK-NO-MERGE
; RUN: llc < %s -mtriple=arm-macho -arm-global-merge                                 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-MERGE
; RUN: llc < %s -mtriple=arm-eabi  -arm-global-merge -relocation-model=pic           | FileCheck %s --check-prefixes=CHECK,CHECK-NO-MERGE

@x = global i32 0, align 4
@y = global i32 0, align 4
@z = global i32 0, align 4

define void @f1(i32 %a1, i32 %a2) {
;CHECK:          f1:
;CHECK:          ldr {{r[0-9]+}}, [[LABEL1:\.?LCPI[0-9]+_[0-9]]]
;CHECK:          [[LABEL1]]:
;CHECK-MERGE:    .long .L_MergedGlobals
;CHECK-NO-MERGE: .long {{_?x}}
  store i32 %a1, i32* @x, align 4
  store i32 %a2, i32* @y, align 4
  ret void
}

define void @g1(i32 %a1, i32 %a2) {
;CHECK:          g1:
;CHECK:          ldr {{r[0-9]+}}, [[LABEL2:\.?LCPI[0-9]+_[0-9]]]
;CHECK:          [[LABEL2]]:
;CHECK-MERGE:    .long .L_MergedGlobals
;CHECK-NO-MERGE: .long {{_?y}}
  store i32 %a1, i32* @y, align 4
  store i32 %a2, i32* @z, align 4
  ret void
}

;CHECK-NO-MERGE-NOT: .globl .L_MergedGlobals

;CHECK-MERGE:	.type	.L_MergedGlobals,%object
;CHECK-MERGE:	.local	.L_MergedGlobals
;CHECK-MERGE:	.comm	.L_MergedGlobals,12,4

;CHECK-MERGE:	.globl	x
;CHECK-MERGE: x = .L_MergedGlobals
;CHECK-MERGE: .size x, 4
;CHECK-MERGE:	.globl	y
;CHECK-MERGE: y = .L_MergedGlobals+4
;CHECK-MERGE: .size y, 4
;CHECK-MERGE:	.globl	z
;CHECK-MERGE: z = .L_MergedGlobals+8
;CHECK-MERGE: .size z, 4
