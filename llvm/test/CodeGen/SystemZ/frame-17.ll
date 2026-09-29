; Test spilling of FPRs.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; We need to save and restore 8 of the 16 FPRs and allocate an additional
; 4-byte spill slot, rounded to 8 bytes.  The frame size should be exactly
; 160 + 8 * 8 = 232.
define void @f1(ptr %ptr) {
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
  %l0 = load volatile float, ptr %ptr
  %l1 = load volatile float, ptr %ptr
  %l2 = load volatile float, ptr %ptr
  %l3 = load volatile float, ptr %ptr
  %l4 = load volatile float, ptr %ptr
  %l5 = load volatile float, ptr %ptr
  %l6 = load volatile float, ptr %ptr
  %l7 = load volatile float, ptr %ptr
  %l8 = load volatile float, ptr %ptr
  %l9 = load volatile float, ptr %ptr
  %l10 = load volatile float, ptr %ptr
  %l11 = load volatile float, ptr %ptr
  %l12 = load volatile float, ptr %ptr
  %l13 = load volatile float, ptr %ptr
  %l14 = load volatile float, ptr %ptr
  %l15 = load volatile float, ptr %ptr
  %lx = load volatile float, ptr %ptr
  store volatile float %lx, ptr %ptr
  store volatile float %l15, ptr %ptr
  store volatile float %l14, ptr %ptr
  store volatile float %l13, ptr %ptr
  store volatile float %l12, ptr %ptr
  store volatile float %l11, ptr %ptr
  store volatile float %l10, ptr %ptr
  store volatile float %l9, ptr %ptr
  store volatile float %l8, ptr %ptr
  store volatile float %l7, ptr %ptr
  store volatile float %l6, ptr %ptr
  store volatile float %l5, ptr %ptr
  store volatile float %l4, ptr %ptr
  store volatile float %l3, ptr %ptr
  store volatile float %l2, ptr %ptr
  store volatile float %l1, ptr %ptr
  store volatile float %l0, ptr %ptr
  ret void
}

; Same for doubles, except that the full spill slot is used.
define void @f2(ptr %ptr) {
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
  %l0 = load volatile double, ptr %ptr
  %l1 = load volatile double, ptr %ptr
  %l2 = load volatile double, ptr %ptr
  %l3 = load volatile double, ptr %ptr
  %l4 = load volatile double, ptr %ptr
  %l5 = load volatile double, ptr %ptr
  %l6 = load volatile double, ptr %ptr
  %l7 = load volatile double, ptr %ptr
  %l8 = load volatile double, ptr %ptr
  %l9 = load volatile double, ptr %ptr
  %l10 = load volatile double, ptr %ptr
  %l11 = load volatile double, ptr %ptr
  %l12 = load volatile double, ptr %ptr
  %l13 = load volatile double, ptr %ptr
  %l14 = load volatile double, ptr %ptr
  %l15 = load volatile double, ptr %ptr
  %lx = load volatile double, ptr %ptr
  store volatile double %lx, ptr %ptr
  store volatile double %l15, ptr %ptr
  store volatile double %l14, ptr %ptr
  store volatile double %l13, ptr %ptr
  store volatile double %l12, ptr %ptr
  store volatile double %l11, ptr %ptr
  store volatile double %l10, ptr %ptr
  store volatile double %l9, ptr %ptr
  store volatile double %l8, ptr %ptr
  store volatile double %l7, ptr %ptr
  store volatile double %l6, ptr %ptr
  store volatile double %l5, ptr %ptr
  store volatile double %l4, ptr %ptr
  store volatile double %l3, ptr %ptr
  store volatile double %l2, ptr %ptr
  store volatile double %l1, ptr %ptr
  store volatile double %l0, ptr %ptr
  ret void
}

; The long double case needs a 16-byte spill slot.
define void @f3(ptr %ptr) {
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
  %l0 = load volatile fp128, ptr %ptr
  %l1 = load volatile fp128, ptr %ptr
  %l4 = load volatile fp128, ptr %ptr
  %l5 = load volatile fp128, ptr %ptr
  %l8 = load volatile fp128, ptr %ptr
  %l9 = load volatile fp128, ptr %ptr
  %l12 = load volatile fp128, ptr %ptr
  %l13 = load volatile fp128, ptr %ptr
  %lx = load volatile fp128, ptr %ptr
  store volatile fp128 %lx, ptr %ptr
  store volatile fp128 %l13, ptr %ptr
  store volatile fp128 %l12, ptr %ptr
  store volatile fp128 %l9, ptr %ptr
  store volatile fp128 %l8, ptr %ptr
  store volatile fp128 %l5, ptr %ptr
  store volatile fp128 %l4, ptr %ptr
  store volatile fp128 %l1, ptr %ptr
  store volatile fp128 %l0, ptr %ptr
  ret void
}
