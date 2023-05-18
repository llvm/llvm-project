; RUN: llc -verify-machineinstrs -o - %s -mtriple=aarch64-none-linux-gnu -fp-contract=fast | FileCheck %s
; RUN: llc -verify-machineinstrs -o - %s -mtriple=arm64-apple-ios7.0 | FileCheck %s -check-prefix=CHECK-NOFAST

declare float @llvm.fma.f32(float, float, float)
declare double @llvm.fma.f64(double, double, double)

define float @test_fmadd(float %a, float %b, float %c) {
; CHECK-LABEL: test_fmadd:
; CHECK-NOFAST-LABEL: test_fmadd:
  %val = call float @llvm.fma.f32(float %a, float %b, float %c)
; CHECK: fmadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK-NOFAST: fmadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  ret float %val
}

define float @test_fmsub(float %a, float %b, float %c) {
; CHECK-LABEL: test_fmsub:
; CHECK-NOFAST-LABEL: test_fmsub:
  %nega = fsub float -0.0, %a
  %val = call float @llvm.fma.f32(float %nega, float %b, float %c)
; CHECK: fmsub {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK-NOFAST: fmsub {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  ret float %val
}

define float @test_fnmadd(float %a, float %b, float %c) {
; CHECK-LABEL: test_fnmadd:
; CHECK-NOFAST-LABEL: test_fnmadd:
  %nega = fsub float -0.0, %a
  %negc = fsub float -0.0, %c
  %val = call float @llvm.fma.f32(float %nega, float %b, float %negc)
; CHECK: fnmadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK-NOFAST: fnmadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  ret float %val
}

define float @test_fnmsub(float %a, float %b, float %c) {
; CHECK-LABEL: test_fnmsub:
; CHECK-NOFAST-LABEL: test_fnmsub:
  %negc = fsub float -0.0, %c
  %val = call float @llvm.fma.f32(float %a, float %b, float %negc)
; CHECK: fnmsub {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK-NOFAST: fnmsub {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  ret float %val
}

define double @testd_fmadd(double %a, double %b, double %c) {
; CHECK-LABEL: testd_fmadd:
; CHECK-NOFAST-LABEL: testd_fmadd:
  %val = call double @llvm.fma.f64(double %a, double %b, double %c)
; CHECK: fmadd {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
; CHECK-NOFAST: fmadd {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  ret double %val
}

define double @testd_fmsub(double %a, double %b, double %c) {
; CHECK-LABEL: testd_fmsub:
; CHECK-NOFAST-LABEL: testd_fmsub:
  %nega = fsub double -0.0, %a
  %val = call double @llvm.fma.f64(double %nega, double %b, double %c)
; CHECK: fmsub {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
; CHECK-NOFAST: fmsub {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  ret double %val
}

define double @testd_fnmadd(double %a, double %b, double %c) {
; CHECK-LABEL: testd_fnmadd:
; CHECK-NOFAST-LABEL: testd_fnmadd:
  %nega = fsub double -0.0, %a
  %negc = fsub double -0.0, %c
  %val = call double @llvm.fma.f64(double %nega, double %b, double %negc)
; CHECK: fnmadd {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
; CHECK-NOFAST: fnmadd {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  ret double %val
}

define double @testd_fnmsub(double %a, double %b, double %c) {
; CHECK-LABEL: testd_fnmsub:
; CHECK-NOFAST-LABEL: testd_fnmsub:
  %negc = fsub double -0.0, %c
  %val = call double @llvm.fma.f64(double %a, double %b, double %negc)
; CHECK: fnmsub {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
; CHECK-NOFAST: fnmsub {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  ret double %val
}

define float @test_fmadd_unfused(float %a, float %b, float %c) {
; CHECK-LABEL: test_fmadd_unfused:
; CHECK-NOFAST-LABEL: test_fmadd_unfused:
  %prod = fmul float %b, %c
  %sum = fadd float %a, %prod
; CHECK: fmadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK-NOFAST-NOT: fmadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK-NOFAST: fmul {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK-NOFAST: fadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  ret float %sum
}

define float @test_fmsub_unfused(float %a, float %b, float %c) {
; CHECK-LABEL: test_fmsub_unfused:
; CHECK-NOFAST-LABEL: test_fmsub_unfused:
  %prod = fmul float %b, %c
  %diff = fsub float %a, %prod
; CHECK: fmsub {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK-NOFAST-NOT: fmsub {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK-NOFAST: fmul {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK-NOFAST: fsub {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  ret float %diff
}

define float @test_fnmadd_unfused(float %a, float %b, float %c) {
; CHECK-LABEL: test_fnmadd_unfused:
; CHECK-NOFAST-LABEL: test_fnmadd_unfused:
  %nega = fsub float -0.0, %a
  %prod = fmul float %b, %c
  %diff = fsub float %nega, %prod
; CHECK: fnmadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK-NOFAST-NOT: fnmadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK-NOFAST: fmul {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK-NOFAST: fsub {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK-NOFAST: ret
  ret float %diff
}

define float @test_fnmsub_unfused(float %a, float %b, float %c) {
; CHECK-LABEL: test_fnmsub_unfused:
; CHECK-NOFAST-LABEL: test_fnmsub_unfused:
  %nega = fsub float -0.0, %a
  %prod = fmul float %b, %c
  %sum = fadd float %nega, %prod
; CHECK: fnmsub {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK-NOFAST-NOT: fnmsub {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK-NOFAST: fmul {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK-NOFAST: fsub {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  ret float %sum
}

; Another set of tests that check for multiply single use

define float @test_fmadd_unfused_su(float %a, float %b, float %c) {
; CHECK-LABEL: test_fmadd_unfused_su:
  %prod = fmul float %b, %c
  %sum = fadd float %a, %prod
  %res = fadd float %sum, %prod
; CHECK-NOT: fmadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK: fmul {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK: fadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK: fadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  ret float %res
}

define float @test_fmsub_unfused_su(float %a, float %b, float %c) {
; CHECK-LABEL: test_fmsub_unfused_su:
  %prod = fmul float %b, %c
  %diff = fsub float %a, %prod
  %res = fsub float %diff, %prod
; CHECK-NOT: fmsub {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK: fmul {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK: fsub {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; CHECK: fsub {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  ret float %res
}

