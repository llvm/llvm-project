; REQUIRES: aarch64-registered-target

;; Test -mattr=help
; RUN: llc -mtriple=aarch64 -mattr=help 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-COMPILE
; RUN: llc -mtriple=aarch64 -mattr=help -o %t.s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-COMPILE
; RUN: llc < %s -mtriple=aarch64 -mattr=help 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-COMPILE
; RUN: llc < %s -mtriple=aarch64 -mattr=help -o %t.s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-COMPILE

;; -mattr=help doesn't have to be first
; RUN: llc -mtriple=aarch64 -mattr=+zcm-fpr64 -mattr=help 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-COMPILE

;; Missing target triple for -mattr=help
; RUN: not llc -mattr=help 2>&1 | FileCheck %s --check-prefixes=CHECK-MISSING-TRIPLE
; RUN: not llc < %s -mattr=help 2>&1 | FileCheck %s --check-prefixes=CHECK-MISSING-TRIPLE

;; Test -mcpu=help
; RUN: llc -mtriple=aarch64 -mcpu=help 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-COMPILE
; RUN: llc -mtriple=aarch64 -mcpu=help -o %t.s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-COMPILE
; RUN: llc < %s -mtriple=aarch64 -mcpu=help 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-COMPILE
; RUN: llc < %s -mtriple=aarch64 -mcpu=help -o %t.s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NO-COMPILE

;; Missing target triple for -mcpu=help
; RUN: not llc -mcpu=help 2>&1 | FileCheck %s --check-prefixes=CHECK-MISSING-TRIPLE
; RUN: not llc < %s -mcpu=help 2>&1 | FileCheck %s --check-prefixes=CHECK-MISSING-TRIPLE

; CHECK: Available CPUs for this target:
; CHECK: Available features for this target:

; CHECK-MISSING-TRIPLE: error: unable to get target for 'unknown', see --version and --triple.

;; To check we dont compile the file
; CHECK-NO-COMPILE-NOT: foo
define i32 @foo() {
  ret i32 0
}
