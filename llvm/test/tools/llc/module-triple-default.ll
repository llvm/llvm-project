; REQUIRES: aarch64-registered-target
; REQUIRES: default_triple

;; Verify that llc correctly sets the module triple when one is not present.

; RUN: llc -march=aarch64                     -stop-after=finalize-isel -o - %s | FileCheck %s --check-prefix=MARCH
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -stop-after=finalize-isel -o - %s | FileCheck %s --check-prefix=MTRIPLE
; RUN: llc                                    -stop-after=finalize-isel -o - %s | FileCheck %s --check-prefix=DEFAULT

; MARCH: target triple = "aarch64-{{.*}}-{{.*}}"
; MTRIPLE: target triple = "aarch64-unknown-linux-gnu"
; DEFAULT: target triple = "{{.*}}-{{.*}}-{{.*}}"

define void @f() { ret void }
