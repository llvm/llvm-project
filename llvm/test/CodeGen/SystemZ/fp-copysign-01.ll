; Test copysign operations.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare float @copysignf(float, float) readnone
declare double @copysign(double, double) readnone
; FIXME: not really the correct prototype for SystemZ.
declare fp128 @copysignl(fp128, fp128) readnone

; Test f32 copies in which the sign comes from an f32.
define float @f1(float %a, float %b) {
; CHECK-LABEL: f1:
; CHECK-NOT: %f2
; CHECK: cpsdr %f0, %f0, %f2
; CHECK: br %r14
  %res = call float @copysignf(float %a, float %b) readnone
  ret float %res
}

; Test f32 copies in which the sign comes from an f64.
define float @f2(float %a, double %bd) {
; CHECK-LABEL: f2:
; CHECK-NOT: %f2
; CHECK: cpsdr %f0, %f0, %f2
; CHECK: br %r14
  %b = fptrunc double %bd to float
  %res = call float @copysignf(float %a, float %b) readnone
  ret float %res
}

; Test f32 copies in which the sign comes from an f128.
define float @f3(float %a, fp128 *%bptr) {
; CHECK-LABEL: f3:
; CHECK: ld [[BHIGH:%f[0-7]]], 0(%r2)
; CHECK: ld [[BLOW:%f[0-7]]], 8(%r2)
; CHECK: cpsdr %f0, %f0, [[BHIGH]]
; CHECK: br %r14
  %bl = load volatile fp128 , fp128 *%bptr
  %b = fptrunc fp128 %bl to float
  %res = call float @copysignf(float %a, float %b) readnone
  ret float %res
}

; Test f64 copies in which the sign comes from an f32.
define double @f4(double %a, float %bf) {
; CHECK-LABEL: f4:
; CHECK-NOT: %f2
; CHECK: cpsdr %f0, %f0, %f2
; CHECK: br %r14
  %b = fpext float %bf to double
  %res = call double @copysign(double %a, double %b) readnone
  ret double %res
}

; Test f64 copies in which the sign comes from an f64.
define double @f5(double %a, double %b) {
; CHECK-LABEL: f5:
; CHECK-NOT: %f2
; CHECK: cpsdr %f0, %f0, %f2
; CHECK: br %r14
  %res = call double @copysign(double %a, double %b) readnone
  ret double %res
}

; Test f64 copies in which the sign comes from an f128.
define double @f6(double %a, fp128 *%bptr) {
; CHECK-LABEL: f6:
; CHECK: ld [[BHIGH:%f[0-7]]], 0(%r2)
; CHECK: ld [[BLOW:%f[0-7]]], 8(%r2)
; CHECK: cpsdr %f0, %f0, [[BHIGH]]
; CHECK: br %r14
  %bl = load volatile fp128 , fp128 *%bptr
  %b = fptrunc fp128 %bl to double
  %res = call double @copysign(double %a, double %b) readnone
  ret double %res
}

; Test f128 copies in which the sign comes from an f32.  We shouldn't
; need any register shuffling here; %a should be tied to %c, with CPSDR
; just changing the high register.
define void @f7(fp128 *%cptr, fp128 *%aptr, float %bf) {
; CHECK-LABEL: f7:
; CHECK: ld [[AHIGH:%f[0-7]]], 0(%r3)
; CHECK: ld [[ALOW:%f[0-7]]], 8(%r3)
; CHECK: cpsdr [[AHIGH]], [[AHIGH]], %f0
; CHECK: std [[AHIGH]], 0(%r2)
; CHECK: std [[ALOW]], 8(%r2)
; CHECK: br %r14
  %a = load volatile fp128 , fp128 *%aptr
  %b = fpext float %bf to fp128
  %c = call fp128 @copysignl(fp128 %a, fp128 %b) readnone
  store fp128 %c, fp128 *%cptr
  ret void
}

; As above, but the sign comes from an f64.
define void @f8(fp128 *%cptr, fp128 *%aptr, double %bd) {
; CHECK-LABEL: f8:
; CHECK: ld [[AHIGH:%f[0-7]]], 0(%r3)
; CHECK: ld [[ALOW:%f[0-7]]], 8(%r3)
; CHECK: cpsdr [[AHIGH]], [[AHIGH]], %f0
; CHECK: std [[AHIGH]], 0(%r2)
; CHECK: std [[ALOW]], 8(%r2)
; CHECK: br %r14
  %a = load volatile fp128 , fp128 *%aptr
  %b = fpext double %bd to fp128
  %c = call fp128 @copysignl(fp128 %a, fp128 %b) readnone
  store fp128 %c, fp128 *%cptr
  ret void
}

; As above, but the sign comes from an f128.  Don't require the low part
; of %b to be loaded, since it isn't used.
define void @f9(fp128 *%cptr, fp128 *%aptr, fp128 *%bptr) {
; CHECK-LABEL: f9:
; CHECK: ld [[AHIGH:%f[0-7]]], 0(%r3)
; CHECK: ld [[ALOW:%f[0-7]]], 8(%r3)
; CHECK: ld [[BHIGH:%f[0-7]]], 0(%r4)
; CHECK: cpsdr [[AHIGH]], [[AHIGH]], [[BHIGH]]
; CHECK: std [[AHIGH]], 0(%r2)
; CHECK: std [[ALOW]], 8(%r2)
; CHECK: br %r14
  %a = load volatile fp128 , fp128 *%aptr
  %b = load volatile fp128 , fp128 *%bptr
  %c = call fp128 @copysignl(fp128 %a, fp128 %b) readnone
  store fp128 %c, fp128 *%cptr
  ret void
}
