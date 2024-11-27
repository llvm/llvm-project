; Test f128 copysign operations on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare half @copysignh(half, half) readnone
declare float @copysignf(float, float) readnone
declare double @copysign(double, double) readnone
; FIXME: not really the correct prototype for SystemZ.
declare fp128 @copysignl(fp128, fp128) readnone

; Test f16 copies in which the sign comes from an f128.
define half @f0(half %a, ptr %bptr) {
; CHECK-LABEL: f0:
; CHECK: vl %v[[REG:[0-9]+]], 0(%r2)
; CHECK: brasl %r14, __trunctfhf2@PLT
; CHECK: brasl %r14, copysignh@PLT
; CHECK: br %r14
  %bl = load volatile fp128, ptr %bptr
  %b = fptrunc fp128 %bl to half
  %res = call half @copysignh(half %a, half %b) readnone
  ret half %res
}

; Test f32 copies in which the sign comes from an f128.
define float @f1(float %a, ptr %bptr) {
; CHECK-LABEL: f1:
; CHECK: vl %v[[REG:[0-9]+]], 0(%r2)
; CHECK: cpsdr %f0, %f[[REG]], %f0
; CHECK: br %r14
  %bl = load volatile fp128, ptr %bptr
  %b = fptrunc fp128 %bl to float
  %res = call float @copysignf(float %a, float %b) readnone
  ret float %res
}

; Test f64 copies in which the sign comes from an f128.
define double @f2(double %a, ptr %bptr) {
; CHECK-LABEL: f2:
; CHECK: vl %v[[REG:[0-9]+]], 0(%r2)
; CHECK: cpsdr %f0, %f[[REG]], %f0
; CHECK: br %r14
  %bl = load volatile fp128, ptr %bptr
  %b = fptrunc fp128 %bl to double
  %res = call double @copysign(double %a, double %b) readnone
  ret double %res
}

; Test f128 copies in which the sign comes from an f16.
define void @f7_half(ptr %cptr, ptr %aptr, half %bh) {
; CHECK-LABEL: f7_half:
; CHECK: vl [[REG1:%v[0-7]+]], 0(%r3)
; CHECK: vsteh   %v0, 164(%r15), 0
; CHECK: tm      164(%r15), 128
; CHECK: wflnxb [[REG2:%v[0-9]+]], [[REG1]]
; CHECK: wflpxb [[REG2]], [[REG1]]
  %a = load volatile fp128, ptr %aptr
  %b = fpext half %bh to fp128
  %c = call fp128 @copysignl(fp128 %a, fp128 %b) readnone
  store fp128 %c, ptr %cptr
  ret void
}

; Test f128 copies in which the sign comes from an f32.
define void @f7(ptr %cptr, ptr %aptr, float %bf) {
; CHECK-LABEL: f7:
; CHECK: vl [[REG1:%v[0-7]+]], 0(%r3)
; CHECK: tmlh
; CHECK: wflnxb [[REG2:%v[0-9]+]], [[REG1]]
; CHECK: wflpxb [[REG2]], [[REG1]]
; CHECK: vst [[REG2]], 0(%r2)
; CHECK: br %r14
  %a = load volatile fp128, ptr %aptr
  %b = fpext float %bf to fp128
  %c = call fp128 @copysignl(fp128 %a, fp128 %b) readnone
  store fp128 %c, ptr %cptr
  ret void
}

; As above, but the sign comes from an f64.
define void @f8(ptr %cptr, ptr %aptr, double %bd) {
; CHECK-LABEL: f8:
; CHECK: vl [[REG1:%v[0-7]+]], 0(%r3)
; CHECK: tmhh
; CHECK: wflnxb [[REG2:%v[0-9]+]], [[REG1]]
; CHECK: wflpxb [[REG2]], [[REG1]]
; CHECK: vst [[REG2]], 0(%r2)
; CHECK: br %r14
  %a = load volatile fp128, ptr %aptr
  %b = fpext double %bd to fp128
  %c = call fp128 @copysignl(fp128 %a, fp128 %b) readnone
  store fp128 %c, ptr %cptr
  ret void
}

; As above, but the sign comes from an f128.
define void @f9(ptr %cptr, ptr %aptr, ptr %bptr) {
; CHECK-LABEL: f9:
; CHECK: vl [[REG1:%v[0-7]+]], 0(%r3)
; CHECK: vl [[REG2:%v[0-7]+]], 0(%r4)
; CHECK: tm
; CHECK: wflnxb [[REG1]], [[REG1]]
; CHECK: wflpxb [[REG1]], [[REG1]]
; CHECK: vst [[REG1]], 0(%r2)
; CHECK: br %r14
  %a = load volatile fp128, ptr %aptr
  %b = load volatile fp128, ptr %bptr
  %c = call fp128 @copysignl(fp128 %a, fp128 %b) readnone
  store fp128 %c, ptr %cptr
  ret void
}
