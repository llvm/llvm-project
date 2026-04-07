; REQUIRES: aarch64-registered-target
; REQUIRES: default_triple

;; Test -mattr=help
; RUN: opt -mtriple=aarch64 -mattr=help 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-OPTIMIZE
; RUN: opt -mtriple=aarch64 -mattr=help -o %t.s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-OPTIMIZE
; RUN: opt < %s -mtriple=aarch64 -mattr=help 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-OPTIMIZE
; RUN: opt < %s -mtriple=aarch64 -mattr=help -o %t.s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-OPTIMIZE

;; -mattr=help doesn't have to be first
; RUN: opt -mtriple=aarch64 -mattr=+zcm-fpr64 -mattr=help 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-OPTIMIZE

;; Using default target triple for -mattr=help
; RUN: opt -mattr=help 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-OPTIMIZE
; RUN: opt < %s -mattr=help 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-OPTIMIZE

;; Test -mcpu=help
; RUN: opt -mtriple=aarch64 -mcpu=help 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-OPTIMIZE
; RUN: opt -mtriple=aarch64 -mcpu=help -o %t.s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-OPTIMIZE
; RUN: opt < %s -mtriple=aarch64 -mcpu=help 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-OPTIMIZE
; RUN: opt < %s -mtriple=aarch64 -mcpu=help -o %t.s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-OPTIMIZE

;; Using default target triple for -mcpu=help
; RUN: opt -mcpu=help 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-OPTIMIZE
; RUN: opt < %s -mcpu=help 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-OPTIMIZE

; CHECK: Available CPUs for this target:
; CHECK: Available features for this target:

;; To check we dont compile the file
; CHECK-NO-OPTIMIZE-NOT: foo
define i32 @foo() {
  ret i32 0
}
