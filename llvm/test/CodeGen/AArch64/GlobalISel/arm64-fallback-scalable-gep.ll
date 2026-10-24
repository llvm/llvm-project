; RUN: llc -mtriple=aarch64 -mattr=+sve -O0 -global-isel -global-isel-abort=2 %s -o - 2>&1 | FileCheck %s

; CHECK: warning: Instruction selection used fallback path for gep_scalable
; CHECK-LABEL: gep_scalable:
; CHECK: rdvl x8, #2
; CHECK: add x0, x0, x8
define ptr @gep_scalable(ptr %p) {
  %q = getelementptr <vscale x 16 x i8>, ptr %p, i64 2
  ret ptr %q
}