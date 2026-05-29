; Negative test: a kernel with no `fneg` operand is not modified by the
; lower-fneg pass (no spurious `fsub -0.0, x` introduced anywhere).
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK-NOT: fsub float -0.0
; CHECK: fadd float %a, %b
; CHECK: ret void

define void @kernel(float %a, float %b, ptr %out) {
entry:
  %s = fadd float %a, %b
  store float %s, ptr %out
  ret void
}
