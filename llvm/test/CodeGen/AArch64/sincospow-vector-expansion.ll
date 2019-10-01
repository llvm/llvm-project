; RUN: llc -o - %s -verify-machineinstrs -mtriple=aarch64-linux-gnu -mattr=+neon | FileCheck %s
; RUN: llc -o - %s -verify-machineinstrs -mtriple=aarch64-linux-gnu \
; RUN:     -mattr=+neon -global-isel -global-isel-abort=2 \
; RUN:     -pass-remarks-missed=gisel* \
; RUN:     2>&1 | FileCheck %s  --check-prefixes=FALLBACK,GISEL

; FALLBACK-NOT: remark{{.*}}test_cos_v2f32
define <2 x float> @test_cos_v2f64(<2 x double> %v1) {
; CHECK-LABEL: test_cos_v2f64:
; CHECK: bl cos
; CHECK: bl cos
; GISEL-LABEL: test_cos_v2f64:
; GISEL: bl cos
; GISEL: bl cos
  %1 = call <2 x double> @llvm.cos.v2f64(<2 x double> %v1)
  %2 = fptrunc <2 x double> %1 to <2 x float>
  ret <2 x float> %2
}

; FALLBACK-NOT: remark{{.*}}test_sin_v2f32
define <2 x float> @test_sin_v2f64(<2 x double> %v1) {
; CHECK-LABEL: test_sin_v2f64:
; CHECK: bl sin
; CHECK: bl sin
; GISEL-LABEL: test_sin_v2f64:
; GISEL: bl sin
; GISEL: bl sin
  %1 = call <2 x double> @llvm.sin.v2f64(<2 x double> %v1)
  %2 = fptrunc <2 x double> %1 to <2 x float>
  ret <2 x float> %2
}

define <2 x float> @test_pow_v2f64(<2 x double> %v1, <2 x double> %v2) {
; CHECK-LABEL: test_pow_v2f64:
; CHECK: bl pow
; CHECK: bl pow
  %1 = call <2 x double> @llvm.pow.v2f64(<2 x double> %v1, <2 x double> %v2)
  %2 = fptrunc <2 x double> %1 to <2 x float>
  ret <2 x float> %2
}

declare <2 x double> @llvm.cos.v2f64(<2 x double>)
declare <2 x double> @llvm.sin.v2f64(<2 x double>)
declare <2 x double> @llvm.pow.v2f64(<2 x double>, <2 x double>)

; FALLBACK-NOT: remark{{.*}}test_cos_v2f32
define <2 x float> @test_cos_v2f32(<2 x float> %v1) {
; CHECK-LABEL: test_cos_v2f32:
; CHECK: bl cos
; CHECK: bl cos
; GISEL-LABEL: test_cos_v2f32:
; GISEL: bl cos
; GISEL: bl cos
  %1 = call <2 x float> @llvm.cos.v2f32(<2 x float> %v1)
  ret <2 x float> %1
}

; FALLBACK-NOT: remark{{.*}}test_sin_v2f32
define <2 x float> @test_sin_v2f32(<2 x float> %v1) {
; CHECK-LABEL: test_sin_v2f32:
; CHECK: bl sin
; CHECK: bl sin
; GISEL-LABEL: test_sin_v2f32:
; GISEL: bl sin
; GISEL: bl sin
  %1 = call <2 x float> @llvm.sin.v2f32(<2 x float> %v1)
  ret <2 x float> %1
}

define <2 x float> @test_pow_v2f32(<2 x float> %v1, <2 x float> %v2) {
; CHECK-LABEL: test_pow_v2f32:
; CHECK: bl pow
; CHECK: bl pow
  %1 = call <2 x float> @llvm.pow.v2f32(<2 x float> %v1, <2 x float> %v2)
  ret <2 x float> %1
}

declare <2 x float> @llvm.cos.v2f32(<2 x float>)
declare <2 x float> @llvm.sin.v2f32(<2 x float>)
declare <2 x float> @llvm.pow.v2f32(<2 x float>, <2 x float>)

; FALLBACK-NOT: remark{{.*}}test_cos_v4f32
define <4 x float> @test_cos_v4f32(<4 x float> %v1) {
; CHECK-LABEL: test_cos_v4f32:
; CHECK: bl cos
; CHECK: bl cos
; CHECK: bl cos
; CHECK: bl cos
; GISEL-LABEL: test_cos_v4f32:
; GISEL: bl cos
; GISEL: bl cos
; GISEL: bl cos
; GISEL: bl cos
  %1 = call <4 x float> @llvm.cos.v4f32(<4 x float> %v1)
  ret <4 x float> %1
}

; FALLBACK-NOT: remark{{.*}}test_sin_v4f32
define <4 x float> @test_sin_v4f32(<4 x float> %v1) {
; CHECK-LABEL: test_sin_v4f32:
; CHECK: bl sin
; CHECK: bl sin
; CHECK: bl sin
; CHECK: bl sin
; GISEL-LABEL: test_sin_v4f32:
; GISEL: bl sin
; GISEL: bl sin
; GISEL: bl sin
; GISEL: bl sin
  %1 = call <4 x float> @llvm.sin.v4f32(<4 x float> %v1)
  ret <4 x float> %1
}

define <4 x float> @test_pow_v4f32(<4 x float> %v1, <4 x float> %v2) {
; CHECK-LABEL: test_pow_v4f32:
; CHECK: bl pow
; CHECK: bl pow
; CHECK: bl pow
; CHECK: bl pow
  %1 = call <4 x float> @llvm.pow.v4f32(<4 x float> %v1, <4 x float> %v2)
  ret <4 x float> %1
}

declare <4 x float> @llvm.cos.v4f32(<4 x float>)
declare <4 x float> @llvm.sin.v4f32(<4 x float>)
declare <4 x float> @llvm.pow.v4f32(<4 x float>, <4 x float>)
