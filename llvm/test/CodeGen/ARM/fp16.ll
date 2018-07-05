; RUN: llc -mtriple=armv7a--none-eabi < %s | FileCheck --check-prefix=CHECK --check-prefix=CHECK-HARDFLOAT-EABI %s
; RUN: llc -mtriple=armv7a--none-gnueabi < %s | FileCheck --check-prefix=CHECK --check-prefix=CHECK-HARDFLOAT-GNU %s
; RUN: llc -mtriple=armv7a--none-musleabi < %s | FileCheck --check-prefix=CHECK --check-prefix=CHECK-HARDFLOAT-GNU %s
; RUN: llc -mtriple=armv8-eabihf < %s | FileCheck --check-prefix=CHECK --check-prefix=CHECK-ARMV8 %s
; RUN: llc -mtriple=thumbv7m-eabi < %s | FileCheck --check-prefix=CHECK --check-prefix=CHECK-SOFTFLOAT-EABI %s
; RUN: llc -mtriple=thumbv7m-gnueabi < %s | FileCheck --check-prefix=CHECK --check-prefix=CHECK-SOFTFLOAT-GNU %s
; RUN: llc -mtriple=thumbv7m-musleabi < %s | FileCheck --check-prefix=CHECK --check-prefix=CHECK-SOFTFLOAT-GNU %s

;; +fp16 is special: it has f32->f16 (unlike v7), but not f64->f16 (unlike v8).
;; This exposes unsafe-fp-math optimization opportunities; test that.
; RUN: llc -mattr=+vfp3,+fp16 < %s |\
; RUN:   FileCheck --check-prefix=CHECK --check-prefix=CHECK-FP16 --check-prefix=CHECK-FP16-SAFE %s
; RUN: llc -mattr=+vfp3,+fp16 < %s -enable-unsafe-fp-math |\
; RUN:   FileCheck --check-prefix=CHECK --check-prefix=CHECK-FP16 --check-prefix=CHECK-FP16-UNSAFE %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32"
target triple = "armv7---eabihf"

@x = global i16 12902
@y = global i16 0
@z = common global i16 0

define void @foo() nounwind {
; CHECK-LABEL: foo:
entry:
  %0 = load i16, i16* @x, align 2
  %1 = load i16, i16* @y, align 2
  %2 = tail call float @llvm.convert.from.fp16.f32(i16 %0)
; CHECK-HARDFLOAT-EABI: __aeabi_h2f
; CHECK-HARDFLOAT-GNU: __gnu_h2f_ieee
; CHECK-FP16: vcvtb.f32.f16
; CHECK-ARMV8: vcvtb.f32.f16
; CHECK-SOFTFLOAT-EABI: __aeabi_h2f
; CHECK-SOFTFLOAT-GNU: __gnu_h2f_ieee
  %3 = tail call float @llvm.convert.from.fp16.f32(i16 %1)
; CHECK-HARDFLOAT-EABI: __aeabi_h2f
; CHECK-HARDFLOAT-GNU: __gnu_h2f_ieee
; CHECK-FP16: vcvtb.f32.f16
; CHECK-ARMV8: vcvtb.f32.f16
; CHECK-SOFTFLOAT-EABI: __aeabi_h2f
; CHECK-SOFTFLOAT-GNU: __gnu_h2f_ieee
  %4 = fadd float %2, %3
  %5 = tail call i16 @llvm.convert.to.fp16.f32(float %4)
; CHECK-HARDFLOAT-EABI: __aeabi_f2h
; CHECK-HARDFLOAT-GNU: __gnu_f2h_ieee
; CHECK-FP16: vcvtb.f16.f32
; CHECK-ARMV8: vcvtb.f16.f32
; CHECK-SOFTFLOAT-EABI: __aeabi_f2h
; CHECK-SOFTFLOAT-GNU: __gnu_f2h_ieee
  store i16 %5, i16* @x, align 2
  ret void
}

define double @test_from_fp16(i16 %in) {
; CHECK-LABEL: test_from_fp16:
  %val = call double @llvm.convert.from.fp16.f64(i16 %in)
; CHECK-HARDFLOAT-EABI: bl __aeabi_h2f
; CHECK-HARDFLOAT-EABI: vmov [[TMP:s[0-9]+]], r0
; CHECK-HARDFLOAT-EABI: vcvt.f64.f32 {{d[0-9]+}}, [[TMP]]

; CHECK-HARDFLOAT-GNU: bl __gnu_h2f_ieee
; CHECK-HARDFLOAT-GNU: vmov [[TMP:s[0-9]+]], r0
; CHECK-HARDFLOAT-GNU: vcvt.f64.f32 {{d[0-9]+}}, [[TMP]]

; CHECK-FP16: vmov [[TMP16:s[0-9]+]], r0
; CHECK-FP16: vcvtb.f32.f16 [[TMP32:s[0-9]+]], [[TMP16]]
; CHECK-FP16: vcvt.f64.f32 d0, [[TMP32]]

; CHECK-ARMV8: vmov [[TMP:s[0-9]+]], r0
; CHECK-ARMV8: vcvtb.f64.f16 d0, [[TMP]]

; CHECK-SOFTFLOAT-EABI: bl __aeabi_h2f
; CHECK-SOFTFLOAT-EABI: bl __aeabi_f2d

; CHECK-SOFTFLOAT-GNU: bl __gnu_h2f_ieee
; CHECK-SOFTFLOAT-GNU: bl __aeabi_f2d
  ret double %val
}

define i16 @test_to_fp16(double %in) {
; CHECK-LABEL: test_to_fp16:
  %val = call i16 @llvm.convert.to.fp16.f64(double %in)
; CHECK-HARDFLOAT-EABI: bl __aeabi_d2h

; CHECK-HARDFLOAT-GNU: bl __aeabi_d2h

; CHECK-FP16-SAFE: bl __aeabi_d2h

; CHECK-FP16-UNSAFE:      vcvt.f32.f64 s0, d0
; CHECK-FP16-UNSAFE-NEXT: vcvtb.f16.f32 s0, s0

; CHECK-ARMV8: vcvtb.f16.f64 [[TMP:s[0-9]+]], d0
; CHECK-ARMV8: vmov r0, [[TMP]]

; CHECK-SOFTFLOAT-EABI: bl __aeabi_d2h

; CHECK-SOFTFLOAT-GNU: bl __aeabi_d2h
  ret i16 %val
}

declare float @llvm.convert.from.fp16.f32(i16) nounwind readnone
declare double @llvm.convert.from.fp16.f64(i16) nounwind readnone

declare i16 @llvm.convert.to.fp16.f32(float) nounwind readnone
declare i16 @llvm.convert.to.fp16.f64(double) nounwind readnone
