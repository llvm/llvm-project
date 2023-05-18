; Test the Test Data Class instruction logic operation conversion from
; compares.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare float @llvm.fabs.f32(float)
declare double @llvm.fabs.f64(double)
declare fp128 @llvm.fabs.f128(fp128)

; Compare with 0 (unworthy)
define i32 @f1(float %x) {
; CHECK-LABEL: f1
; CHECK-NOT: tceb
; CHECK: ltebr {{%f[0-9]+}}, %f0
; CHECK-NOT: tceb
  %res = fcmp ugt float %x, 0.0
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Compare fabs with 0 (unworthy)
define i32 @f2(float %x) {
; CHECK-LABEL: f2
; CHECK-NOT: tceb
; CHECK: lpebr {{%f[0-9]+}}, %f0
; CHECK-NOT: tceb
  %y = call float @llvm.fabs.f32(float %x)
  %res = fcmp ugt float %y, 0.0
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Compare with inf (unworthy)
define i32 @f3(float %x) {
; CHECK-LABEL: f3
; CHECK-NOT: tceb
; CHECK: ceb %f0, 0(%r{{[0-9]+}})
; CHECK-NOT: tceb
  %res = fcmp ult float %x, 0x7ff0000000000000
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Compare fabs with inf
define i32 @f4(float %x) {
; CHECK-LABEL: f4
; CHECK: tceb %f0, 4047
  %y = call float @llvm.fabs.f32(float %x)
  %res = fcmp ult float %y, 0x7ff0000000000000
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Compare with minnorm (unworthy)
define i32 @f5(float %x) {
; CHECK-LABEL: f5
; CHECK-NOT: tceb
; CHECK: ceb %f0, 0(%r{{[0-9]+}})
; CHECK-NOT: tceb
  %res = fcmp ult float %x, 0x3810000000000000
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Compare fabs with minnorm
define i32 @f6(float %x) {
; CHECK-LABEL: f6
; CHECK: tceb %f0, 3279
  %y = call float @llvm.fabs.f32(float %x)
  %res = fcmp ult float %y, 0x3810000000000000
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Compare fabs with minnorm, unsupported condition
define i32 @f7(float %x) {
; CHECK-LABEL: f7
; CHECK-NOT: tceb
; CHECK: lpdfr [[REG:%f[0-9]+]], %f0
; CHECK: ceb [[REG]], 0(%r{{[0-9]+}})
; CHECK-NOT: tceb
  %y = call float @llvm.fabs.f32(float %x)
  %res = fcmp ugt float %y, 0x3810000000000000
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Compare fabs with unsupported constant
define i32 @f8(float %x) {
; CHECK-LABEL: f8
; CHECK-NOT: tceb
; CHECK: lpdfr [[REG:%f[0-9]+]], %f0
; CHECK: ceb [[REG]], 0(%r{{[0-9]+}})
; CHECK-NOT: tceb
  %y = call float @llvm.fabs.f32(float %x)
  %res = fcmp ult float %y, 0x3ff0000000000000
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Compare fabs with minnorm - double
define i32 @f9(double %x) {
; CHECK-LABEL: f9
; CHECK: tcdb %f0, 3279
  %y = call double @llvm.fabs.f64(double %x)
  %res = fcmp ult double %y, 0x0010000000000000
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Compare fabs with minnorm - long double
define i32 @f10(fp128 %x) {
; CHECK-LABEL: f10
; CHECK: tcxb %f0, 3279
  %y = call fp128 @llvm.fabs.f128(fp128 %x)
  %res = fcmp ult fp128 %y, 0xL00000000000000000001000000000000
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Compare fabs for one with inf - clang's isfinite
define i32 @f11(double %x) {
; CHECK-LABEL: f11
; CHECK: tcdb %f0, 4032
  %y = call double @llvm.fabs.f64(double %x)
  %res = fcmp one double %y, 0x7ff0000000000000
  %xres = zext i1 %res to i32
  ret i32 %xres
}

; Compare fabs for oeq with inf - clang's isinf
define i32 @f12(double %x) {
; CHECK-LABEL: f12
; CHECK: tcdb %f0, 48
  %y = call double @llvm.fabs.f64(double %x)
  %res = fcmp oeq double %y, 0x7ff0000000000000
  %xres = zext i1 %res to i32
  ret i32 %xres
}
