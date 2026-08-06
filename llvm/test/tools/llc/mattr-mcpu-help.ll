; REQUIRES: aarch64-registered-target
; REQUIRES: default_triple

;; Test -mattr=help
; RUN: llc -mtriple=aarch64 -mattr=help 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-COMPILE
; RUN: llc -mtriple=aarch64 -mattr=help -o %t.s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-COMPILE
; RUN: llc < %s -mtriple=aarch64 -mattr=help 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-COMPILE
; RUN: llc < %s -mtriple=aarch64 -mattr=help -o %t.s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-COMPILE

;; -mattr=help doesn't have to be first
; RUN: llc -mtriple=aarch64 -mattr=+zcm-fpr64 -mattr=help 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-COMPILE

;; Using default target triple for -mattr=help
; RUN: llc -mattr=help 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-COMPILE
; RUN: llc < %s -mattr=help 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-COMPILE

;; Test -mcpu=help
; RUN: llc -mtriple=aarch64 -mcpu=help 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-COMPILE
; RUN: llc -mtriple=aarch64 -mcpu=help -o %t.s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-COMPILE
; RUN: llc < %s -mtriple=aarch64 -mcpu=help 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-COMPILE
; RUN: llc < %s -mtriple=aarch64 -mcpu=help -o %t.s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-COMPILE

;; Using default target triple for -mcpu=help
; RUN: llc -mcpu=help 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-COMPILE
; RUN: llc < %s -mcpu=help 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-COMPILE

; CHECK: Available CPUs for this target:
; CHECK: Available features for this target:

;; To check we dont compile the file
; CHECK-NO-COMPILE-NOT: foo
define i32 @foo() {
  ret i32 0
}
