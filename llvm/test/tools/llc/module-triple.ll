; REQUIRES: aarch64-registered-target
; REQUIRES: x86-registered-target

;; Verify that llc correctly sets the module triple when one is present.

; RUN: llc -march=aarch64                   -stop-after=finalize-isel -o - %s | FileCheck %s --check-prefix=MARCH
; RUN: llc -mtriple=aarch64-unknown-unknown -stop-after=finalize-isel -o - %s | FileCheck %s --check-prefix=MTRIPLE
; RUN: llc                                  -stop-after=finalize-isel -o - %s | FileCheck %s --check-prefix=MODULE

; MARCH: target triple = "aarch64-unknown-linux-gnu"
; MTRIPLE: target triple = "aarch64-unknown-unknown"
; MODULE: target triple = "x86_64-unknown-linux-gnu"

target triple = "x86_64-unknown-linux-gnu"

define void @f() { ret void }
