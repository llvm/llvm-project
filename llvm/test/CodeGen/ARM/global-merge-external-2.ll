; RUN: llc < %s -mtriple=arm-eabi  -arm-global-merge                                 | FileCheck %s --check-prefixes=CHECK,CHECK-MERGE
; RUN: llc < %s -mtriple=arm-eabi  -arm-global-merge -global-merge-on-external=true  | FileCheck %s --check-prefixes=CHECK,CHECK-MERGE
; RUN: llc < %s -mtriple=arm-eabi  -arm-global-merge -global-merge-on-external=false | FileCheck %s --check-prefixes=CHECK,CHECK-NO-MERGE
; RUN: llc < %s -mtriple=arm-macho -arm-global-merge                                 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-MERGE
; RUN: llc < %s -mtriple=arm-eabi  -arm-global-merge -relocation-model=pic           | FileCheck %s --check-prefixes=CHECK,CHECK-NO-MERGE
; RUN: llc < %s -mtriple=thumbv7-win32 -arm-global-merge                             | FileCheck %s --check-prefixes=CHECK-WIN32

@x = dso_local global i32 0, align 4
@y = dso_local global i32 0, align 4
@z = internal global i32 1, align 4

define dso_local void @f1(i32 %a1, i32 %a2) {
;CHECK:          f1:
;CHECK:          ldr {{r[0-9]+}}, [[LABEL1:\.?LCPI[0-9]+_[0-9]]]
;CHECK:          [[LABEL1]]:
;CHECK-MERGE:    .long .L_MergedGlobals
;CHECK-NO-MERGE: .long {{_?x|.L_MergedGlobals}}
;CHECK-WIN32:    f1:
;CHECK-WIN32:    movw [[REG1:r[0-9]+]], :lower16:.L_MergedGlobals
;CHECK-WIN32:    movt [[REG1]], :upper16:.L_MergedGlobals
  store i32 %a1, i32* @x, align 4
  store i32 %a2, i32* @y, align 4
  ret void
}

define dso_local void @g1(i32 %a1, i32 %a2) {
;CHECK:          g1:
;CHECK:          ldr {{r[0-9]+}}, [[LABEL2:\.?LCPI[0-9]+_[0-9]]]
;CHECK:          ldr {{r[0-9]+}}, [[LABEL3:\.?LCPI[0-9]+_[0-9]]]
;CHECK:          [[LABEL2]]:
;CHECK-MERGE:    .long {{_?z}}
;CHECK:          [[LABEL3]]:
;CHECK-MERGE:    .long .L_MergedGlobals
;CHECK-NO-MERGE: .long {{_?y|.L_MergedGlobals}}
;CHECK-WIN32:    g1:
;CHECK-WIN32:    movw    [[REG2:r[0-9]+]], :lower16:z
;CHECK-WIN32:    movt    [[REG2]], :upper16:z
;CHECK-WIN32:    movw [[REG3:r[0-9]+]], :lower16:.L_MergedGlobals
;CHECK-WIN32:    movt [[REG3]], :upper16:.L_MergedGlobals
  store i32 %a1, i32* @y, align 4
  store i32 %a2, i32* @z, align 4
  ret void
}

;CHECK-NO-MERGE-NOT: .globl .L_MergedGlobals

;CHECK-MERGE:   .type   .L_MergedGlobals,%object
;CHECK-MERGE:   .local  .L_MergedGlobals
;CHECK-MERGE:   .comm   .L_MergedGlobals,8,4
;CHECK-WIN32:   .lcomm  .L_MergedGlobals,8,4

;CHECK-MERGE:   .globl  x
;CHECK-MERGE: .set x, .L_MergedGlobals
;CHECK-MERGE: .size x, 4
;CHECK-MERGE:   .globl  y
;CHECK-MERGE: .set y, .L_MergedGlobals+4
;CHECK-MERGE: .size y, 4
;CHECK-MERGE-NOT: .set z, .L_MergedGlobals+8


;CHECK-WIN32:   .globl  x
;CHECK-WIN32: .set x, .L_MergedGlobals
;CHECK-WIN32:   .globl  y
;CHECK-WIN32: .set y, .L_MergedGlobals+4
;CHECK-WIN32-NOT: .set z, .L_MergedGlobals+8
