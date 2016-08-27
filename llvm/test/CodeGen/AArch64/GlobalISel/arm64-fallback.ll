; RUN: not llc -O0 -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: llc -O0 -global-isel -global-isel-abort=false -verify-machineinstrs %s -o - 2>&1 | FileCheck %s --check-prefix=FALLBACK
; This file checks that the fallback path to selection dag works.
; The test is fragile in the sense that it must be updated to expose
; something that fails with global-isel.
; When we cannot produce a test case anymore, that means we can remove
; the fallback path.

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-apple-ios"

; ERROR: Unable to lower arguments
; FALLBACK: ldr q0,
; FALLBACK-NEXT: bl ___fixunstfti
define i128 @ABIi128(i128 %arg1) {
  %farg1 =       bitcast i128 %arg1 to fp128 
  %res = fptoui fp128 %farg1 to i128
  ret i128 %res
}
