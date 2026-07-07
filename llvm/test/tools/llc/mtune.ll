; REQUIRES: aarch64-registered-target

;; There shouldn't be a default -mtune.
; RUN: llc < %s -mtriple=aarch64 | FileCheck %s --check-prefixes=CHECK-NOTUNE

; RUN: llc < %s -mtriple=aarch64 -mtune=generic | FileCheck %s --check-prefixes=CHECK-TUNE-GENERIC
; RUN: llc < %s -mtriple=aarch64 -mtune=apple-m5 | FileCheck %s --check-prefixes=CHECK-TUNE-APPLE-M5

;; Check interaction between mcpu and mtune.
; RUN: llc < %s -mtriple=aarch64 -mcpu=apple-m5 | FileCheck %s --check-prefixes=CHECK-TUNE-APPLE-M5
; RUN: llc < %s -mtriple=aarch64 -mcpu=apple-m5 -mtune=generic | FileCheck %s --check-prefixes=CHECK-TUNE-GENERIC

;; Test -mtune=help
; RUN: llc -mtriple=aarch64 -mtune=help 2>&1 | FileCheck %s --check-prefixes=CHECK-TUNE-HELP,CHECK-TUNE-HELP-NO-COMPILE
; RUN: llc -mtriple=aarch64 -mtune=help -o %t.s 2>&1 | FileCheck %s --check-prefixes=CHECK-TUNE-HELP,CHECK-TUNE-HELP-NO-COMPILE
; RUN: llc < %s -mtriple=aarch64 -mtune=help 2>&1 | FileCheck %s --check-prefixes=CHECK-TUNE-HELP,CHECK-TUNE-HELP-NO-COMPILE
; RUN: llc < %s -mtriple=aarch64 -mtune=help -o %t.s 2>&1 | FileCheck %s --check-prefixes=CHECK-TUNE-HELP,CHECK-TUNE-HELP-NO-COMPILE

;; Using default target triple for -mtune=help
; RUN: llc -mtune=help 2>&1 | FileCheck %s --check-prefixes=CHECK-TUNE-HELP,CHECK-TUNE-HELP-NO-COMPILE
; RUN: llc < %s -mtune=help 2>&1 | FileCheck %s --check-prefixes=CHECK-TUNE-HELP,CHECK-TUNE-HELP-NO-COMPILE

; CHECK-TUNE-HELP: Available CPUs for this target:
; CHECK-TUNE-HELP: Available features for this target:

;; To check we dont compile the file
; CHECK-TUNE-HELP-NO-COMPILE-NOT: zero_cycle_regmove_FPR32:

;; A test case that depends on `FeatureZCRegMoveFPR128` tuning feature, to enable -mtune verification
;; through codegen effects. Taken from: llvm/test/CodeGen/AArch64/arm64-zero-cycle-regmove-fpr.ll
define void @zero_cycle_regmove_FPR32(float %a, float %b, float %c, float %d) {
; CHECK-NOTUNE-LABEL: zero_cycle_regmove_FPR32:
; CHECK-NOTUNE:    fmov s0, s2
; CHECK-NOTUNE-NEXT:    fmov s1, s3
; CHECK-NOTUNE-NEXT:    fmov s8, s3
; CHECK-NOTUNE-NEXT:    fmov s9, s2
; CHECK-NOTUNE-NEXT:    bl foo_float
; CHECK-NOTUNE-NEXT:    fmov s0, s9
; CHECK-NOTUNE-NEXT:    fmov s1, s8
; CHECK-NOTUNE-NEXT:    bl foo_float
;
; CHECK-TUNE-GENERIC-LABEL: zero_cycle_regmove_FPR32:
; CHECK-TUNE-GENERIC:    fmov s0, s2
; CHECK-TUNE-GENERIC-NEXT:    fmov s1, s3
; CHECK-TUNE-GENERIC-NEXT:    fmov s8, s3
; CHECK-TUNE-GENERIC-NEXT:    fmov s9, s2
; CHECK-TUNE-GENERIC-NEXT:    bl foo_float
; CHECK-TUNE-GENERIC-NEXT:    fmov s0, s9
; CHECK-TUNE-GENERIC-NEXT:    fmov s1, s8
; CHECK-TUNE-GENERIC-NEXT:    bl foo_float
;
; CHECK-TUNE-APPLE-M5-LABEL: zero_cycle_regmove_FPR32:
; CHECK-TUNE-APPLE-M5:    mov v8.16b, v3.16b
; CHECK-TUNE-APPLE-M5-NEXT:    mov v9.16b, v2.16b
; CHECK-TUNE-APPLE-M5-NEXT:    mov v0.16b, v2.16b
; CHECK-TUNE-APPLE-M5-NEXT:    mov v1.16b, v3.16b
; CHECK-TUNE-APPLE-M5-NEXT:    bl foo_float
; CHECK-TUNE-APPLE-M5-NEXT:    mov v0.16b, v9.16b
; CHECK-TUNE-APPLE-M5-NEXT:    mov v1.16b, v8.16b
; CHECK-TUNE-APPLE-M5-NEXT:    bl foo_float
entry:
  %call = call float @foo_float(float %c, float %d)
  %call1 = call float @foo_float(float %c, float %d)
  unreachable
}

declare float @foo_float(float, float)
