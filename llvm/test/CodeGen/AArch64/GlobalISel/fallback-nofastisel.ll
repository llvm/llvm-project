; RUN: llc -mtriple=aarch64_be-- %s -aarch64-enable-global-isel-at-O=0 -o /dev/null -debug-only=isel -O0 2>&1 | FileCheck %s
; REQUIRES: asserts

; This test uses big endian in order to force an abort since it's not currently supported for GISel.
; The purpose is to check that we don't fall back to FastISel. Checking the pass structure is insufficient
; because the FastISel is set up in the SelectionDAGISel, so it doesn't appear on the pass structure.

; CHECK-NOT: Enabling fast-ise
define void @empty() {
  ret void
}
