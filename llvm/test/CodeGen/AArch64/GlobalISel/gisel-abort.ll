; RUN: llc -march aarch64 -global-isel -global-isel-abort=2 -verify-machineinstrs %s -o - 2>&1 | FileCheck %s

; CHECK-NOT: fallback
; CHECK: empty
define void @empty() {
  ret void
}

