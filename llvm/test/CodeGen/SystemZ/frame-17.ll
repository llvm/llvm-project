; Test spilling of FPRs.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; We need to save and restore 8 of the 16 FPRs and allocate an additional
; 4-byte spill slot, rounded to 8 bytes.  The frame size should be exactly
; 160 + 8 * 8 = 232.
define void @f1(float *%ptr) {
; CHECK-LABEL: f1:
; CHECK: aghi %r15, -232
; CHECK: std %f8, 224(%r15)
; CHECK: std %f9, 216(%r15)
; CHECK: std %f10, 208(%r15)
; CHECK: std %f11, 200(%r15)
; CHECK: std %f12, 192(%r15)
; CHECK: std %f13, 184(%r15)
; CHECK: std %f14, 176(%r15)
; CHECK: std %f15, 168(%r15)
; CHECK-NOT: 160(%r15)
; CHECK: ste [[REGISTER:%f[0-9]+]], 164(%r15)
; CHECK-NOT: 160(%r15)
; CHECK: le [[REGISTER]], 164(%r15)
; CHECK-NOT: 160(%r15)
; CHECK: ld %f8, 224(%r15)
; CHECK: ld %f9, 216(%r15)
; CHECK: ld %f10, 208(%r15)
; CHECK: ld %f11, 200(%r15)
; CHECK: ld %f12, 192(%r15)
; CHECK: ld %f13, 184(%r15)
; CHECK: ld %f14, 176(%r15)
; CHECK: ld %f15, 168(%r15)
; CHECK: aghi %r15, 232
; CHECK: br %r14
  %l0 = load volatile float, float *%ptr
  %l1 = load volatile float, float *%ptr
  %l2 = load volatile float, float *%ptr
  %l3 = load volatile float, float *%ptr
  %l4 = load volatile float, float *%ptr
  %l5 = load volatile float, float *%ptr
  %l6 = load volatile float, float *%ptr
  %l7 = load volatile float, float *%ptr
  %l8 = load volatile float, float *%ptr
  %l9 = load volatile float, float *%ptr
  %l10 = load volatile float, float *%ptr
  %l11 = load volatile float, float *%ptr
  %l12 = load volatile float, float *%ptr
  %l13 = load volatile float, float *%ptr
  %l14 = load volatile float, float *%ptr
  %l15 = load volatile float, float *%ptr
  %lx = load volatile float, float *%ptr
  store volatile float %lx, float *%ptr
  store volatile float %l15, float *%ptr
  store volatile float %l14, float *%ptr
  store volatile float %l13, float *%ptr
  store volatile float %l12, float *%ptr
  store volatile float %l11, float *%ptr
  store volatile float %l10, float *%ptr
  store volatile float %l9, float *%ptr
  store volatile float %l8, float *%ptr
  store volatile float %l7, float *%ptr
  store volatile float %l6, float *%ptr
  store volatile float %l5, float *%ptr
  store volatile float %l4, float *%ptr
  store volatile float %l3, float *%ptr
  store volatile float %l2, float *%ptr
  store volatile float %l1, float *%ptr
  store volatile float %l0, float *%ptr
  ret void
}

; Same for doubles, except that the full spill slot is used.
define void @f2(double *%ptr) {
; CHECK-LABEL: f2:
; CHECK: aghi %r15, -232
; CHECK: std %f8, 224(%r15)
; CHECK: std %f9, 216(%r15)
; CHECK: std %f10, 208(%r15)
; CHECK: std %f11, 200(%r15)
; CHECK: std %f12, 192(%r15)
; CHECK: std %f13, 184(%r15)
; CHECK: std %f14, 176(%r15)
; CHECK: std %f15, 168(%r15)
; CHECK: std [[REGISTER:%f[0-9]+]], 160(%r15)
; CHECK: ld [[REGISTER]], 160(%r15)
; CHECK: ld %f8, 224(%r15)
; CHECK: ld %f9, 216(%r15)
; CHECK: ld %f10, 208(%r15)
; CHECK: ld %f11, 200(%r15)
; CHECK: ld %f12, 192(%r15)
; CHECK: ld %f13, 184(%r15)
; CHECK: ld %f14, 176(%r15)
; CHECK: ld %f15, 168(%r15)
; CHECK: aghi %r15, 232
; CHECK: br %r14
  %l0 = load volatile double, double *%ptr
  %l1 = load volatile double, double *%ptr
  %l2 = load volatile double, double *%ptr
  %l3 = load volatile double, double *%ptr
  %l4 = load volatile double, double *%ptr
  %l5 = load volatile double, double *%ptr
  %l6 = load volatile double, double *%ptr
  %l7 = load volatile double, double *%ptr
  %l8 = load volatile double, double *%ptr
  %l9 = load volatile double, double *%ptr
  %l10 = load volatile double, double *%ptr
  %l11 = load volatile double, double *%ptr
  %l12 = load volatile double, double *%ptr
  %l13 = load volatile double, double *%ptr
  %l14 = load volatile double, double *%ptr
  %l15 = load volatile double, double *%ptr
  %lx = load volatile double, double *%ptr
  store volatile double %lx, double *%ptr
  store volatile double %l15, double *%ptr
  store volatile double %l14, double *%ptr
  store volatile double %l13, double *%ptr
  store volatile double %l12, double *%ptr
  store volatile double %l11, double *%ptr
  store volatile double %l10, double *%ptr
  store volatile double %l9, double *%ptr
  store volatile double %l8, double *%ptr
  store volatile double %l7, double *%ptr
  store volatile double %l6, double *%ptr
  store volatile double %l5, double *%ptr
  store volatile double %l4, double *%ptr
  store volatile double %l3, double *%ptr
  store volatile double %l2, double *%ptr
  store volatile double %l1, double *%ptr
  store volatile double %l0, double *%ptr
  ret void
}

; The long double case needs a 16-byte spill slot.
define void @f3(fp128 *%ptr) {
; CHECK-LABEL: f3:
; CHECK: aghi %r15, -240
; CHECK: std %f8, 232(%r15)
; CHECK: std %f9, 224(%r15)
; CHECK: std %f10, 216(%r15)
; CHECK: std %f11, 208(%r15)
; CHECK: std %f12, 200(%r15)
; CHECK: std %f13, 192(%r15)
; CHECK: std %f14, 184(%r15)
; CHECK: std %f15, 176(%r15)
; CHECK: std [[REGISTER1:%f[0-9]+]], 160(%r15)
; CHECK: std [[REGISTER2:%f[0-9]+]], 168(%r15)
; CHECK: ld [[REGISTER1]], 160(%r15)
; CHECK: ld [[REGISTER2]], 168(%r15)
; CHECK: ld %f8, 232(%r15)
; CHECK: ld %f9, 224(%r15)
; CHECK: ld %f10, 216(%r15)
; CHECK: ld %f11, 208(%r15)
; CHECK: ld %f12, 200(%r15)
; CHECK: ld %f13, 192(%r15)
; CHECK: ld %f14, 184(%r15)
; CHECK: ld %f15, 176(%r15)
; CHECK: aghi %r15, 240
; CHECK: br %r14
  %l0 = load volatile fp128, fp128 *%ptr
  %l1 = load volatile fp128, fp128 *%ptr
  %l4 = load volatile fp128, fp128 *%ptr
  %l5 = load volatile fp128, fp128 *%ptr
  %l8 = load volatile fp128, fp128 *%ptr
  %l9 = load volatile fp128, fp128 *%ptr
  %l12 = load volatile fp128, fp128 *%ptr
  %l13 = load volatile fp128, fp128 *%ptr
  %lx = load volatile fp128, fp128 *%ptr
  store volatile fp128 %lx, fp128 *%ptr
  store volatile fp128 %l13, fp128 *%ptr
  store volatile fp128 %l12, fp128 *%ptr
  store volatile fp128 %l9, fp128 *%ptr
  store volatile fp128 %l8, fp128 *%ptr
  store volatile fp128 %l5, fp128 *%ptr
  store volatile fp128 %l4, fp128 *%ptr
  store volatile fp128 %l1, fp128 *%ptr
  store volatile fp128 %l0, fp128 *%ptr
  ret void
}
