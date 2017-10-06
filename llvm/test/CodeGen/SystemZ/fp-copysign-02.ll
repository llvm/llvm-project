; Test f128 copysign operations on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare float @copysignf(float, float) readnone
declare double @copysign(double, double) readnone
; FIXME: not really the correct prototype for SystemZ.
declare fp128 @copysignl(fp128, fp128) readnone

; Test f32 copies in which the sign comes from an f128.
define float @f1(float %a, fp128 *%bptr) {
; CHECK-LABEL: f1:
; CHECK: vl %v[[REG:[0-9]+]], 0(%r2)
; CHECK: cpsdr %f0, %f[[REG]], %f0
; CHECK: br %r14
  %bl = load volatile fp128, fp128 *%bptr
  %b = fptrunc fp128 %bl to float
  %res = call float @copysignf(float %a, float %b) readnone
  ret float %res
}

; Test f64 copies in which the sign comes from an f128.
define double @f2(double %a, fp128 *%bptr) {
; CHECK-LABEL: f2:
; CHECK: vl %v[[REG:[0-9]+]], 0(%r2)
; CHECK: cpsdr %f0, %f[[REG]], %f0
; CHECK: br %r14
  %bl = load volatile fp128, fp128 *%bptr
  %b = fptrunc fp128 %bl to double
  %res = call double @copysign(double %a, double %b) readnone
  ret double %res
}

; Test f128 copies in which the sign comes from an f32.
define void @f7(fp128 *%cptr, fp128 *%aptr, float %bf) {
; CHECK-LABEL: f7:
; CHECK: vl [[REG1:%v[0-7]+]], 0(%r3)
; CHECK: tmlh
; CHECK: wflnxb [[REG2:%v[0-9]+]], [[REG1]]
; CHECK: wflpxb [[REG2]], [[REG1]]
; CHECK: vst [[REG2]], 0(%r2)
; CHECK: br %r14
  %a = load volatile fp128, fp128 *%aptr
  %b = fpext float %bf to fp128
  %c = call fp128 @copysignl(fp128 %a, fp128 %b) readnone
  store fp128 %c, fp128 *%cptr
  ret void
}

; As above, but the sign comes from an f64.
define void @f8(fp128 *%cptr, fp128 *%aptr, double %bd) {
; CHECK-LABEL: f8:
; CHECK: vl [[REG1:%v[0-7]+]], 0(%r3)
; CHECK: tmhh
; CHECK: wflnxb [[REG2:%v[0-9]+]], [[REG1]]
; CHECK: wflpxb [[REG2]], [[REG1]]
; CHECK: vst [[REG2]], 0(%r2)
; CHECK: br %r14
  %a = load volatile fp128, fp128 *%aptr
  %b = fpext double %bd to fp128
  %c = call fp128 @copysignl(fp128 %a, fp128 %b) readnone
  store fp128 %c, fp128 *%cptr
  ret void
}

; As above, but the sign comes from an f128.
define void @f9(fp128 *%cptr, fp128 *%aptr, fp128 *%bptr) {
; CHECK-LABEL: f9:
; CHECK: vl [[REG1:%v[0-7]+]], 0(%r3)
; CHECK: vl [[REG2:%v[0-7]+]], 0(%r4)
; CHECK: tm
; CHECK: wflnxb [[REG1]], [[REG1]]
; CHECK: wflpxb [[REG1]], [[REG1]]
; CHECK: vst [[REG1]], 0(%r2)
; CHECK: br %r14
  %a = load volatile fp128, fp128 *%aptr
  %b = load volatile fp128, fp128 *%bptr
  %c = call fp128 @copysignl(fp128 %a, fp128 %b) readnone
  store fp128 %c, fp128 *%cptr
  ret void
}
