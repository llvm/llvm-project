; Test moves between FPRs.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; Test f32 moves.
define float @f1(float %a, float %b) {
; CHECK-LABEL: f1:
; CHECK: ler %f0, %f2
; CHECK: br %r14
  ret float %b
}

; Test f64 moves.
define double @f2(double %a, double %b) {
; CHECK-LABEL: f2:
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  ret double %b
}

; Test f128 moves.  Since f128s are passed by reference, we need to force
; a copy by other means.
define void @f3(fp128 *%x) {
; CHECK-LABEL: f3:
; CHECK: lxr
; CHECK: axbr
; CHECK: br %r14
  %val = load volatile fp128, fp128 *%x
  %sum = fadd fp128 %val, %val
  store volatile fp128 %sum, fp128 *%x
  store volatile fp128 %val, fp128 *%x
  ret void
}
