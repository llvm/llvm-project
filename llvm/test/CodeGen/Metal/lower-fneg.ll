; The Metal GPU JIT does not support `fneg`; it is lowered to `fsub -0.0, x`.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK-NOT: fneg
; CHECK: fsub float -0.000000e+00, %x
; CHECK: ret void

define void @kernel(float %x, ptr %out) {
entry:
  %n = fneg float %x
  store float %n, ptr %out
  ret void
}
