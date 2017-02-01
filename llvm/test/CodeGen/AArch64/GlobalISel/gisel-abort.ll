; RUN: llc -mtriple=aarch64-unknown-unknown -global-isel -global-isel-abort=2 -verify-machineinstrs %s -o - 2>&1 | FileCheck %s

; CHECK-NOT: fallback
; CHECK: empty
define void @empty() {
  ret void
}

