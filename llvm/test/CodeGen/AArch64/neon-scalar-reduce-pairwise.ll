; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s

declare <1 x i64> @llvm.aarch64.neon.vpadd(<2 x i64>)

define <1 x i64> @test_addp_v1i64(<2 x i64> %a) {
; CHECK: test_addp_v1i64:
        %val = call <1 x i64> @llvm.aarch64.neon.vpadd(<2 x i64> %a)
; CHECK: addp d0, v0.2d
        ret <1 x i64> %val
}

declare <1 x float> @llvm.aarch64.neon.vpfadd(<2 x float>)

define <1 x float> @test_faddp_v1f32(<2 x float> %a) {
; CHECK: test_faddp_v1f32:
        %val = call <1 x float> @llvm.aarch64.neon.vpfadd(<2 x float> %a)
; CHECK: faddp s0, v0.2s
        ret <1 x float> %val
}

declare <1 x double> @llvm.aarch64.neon.vpfaddq(<2 x double>)

define <1 x double> @test_faddp_v1f64(<2 x double> %a) {
; CHECK: test_faddp_v1f64:
        %val = call <1 x double> @llvm.aarch64.neon.vpfaddq(<2 x double> %a)
; CHECK: faddp d0, v0.2d
        ret <1 x double> %val
}


declare <1 x float> @llvm.aarch64.neon.vpmax(<2 x float>)

define <1 x float> @test_fmaxp_v1f32(<2 x float> %a) {
; CHECK: test_fmaxp_v1f32:
        %val = call <1 x float> @llvm.aarch64.neon.vpmax(<2 x float> %a)
; CHECK: fmaxp s0, v0.2s
        ret <1 x float> %val
}

declare <1 x double> @llvm.aarch64.neon.vpmaxq(<2 x double>)

define <1 x double> @test_fmaxp_v1f64(<2 x double> %a) {
; CHECK: test_fmaxp_v1f64:
        %val = call <1 x double> @llvm.aarch64.neon.vpmaxq(<2 x double> %a)
; CHECK: fmaxp d0, v0.2d
        ret <1 x double> %val
}


declare <1 x float> @llvm.aarch64.neon.vpmin(<2 x float>)

define <1 x float> @test_fminp_v1f32(<2 x float> %a) {
; CHECK: test_fminp_v1f32:
        %val = call <1 x float> @llvm.aarch64.neon.vpmin(<2 x float> %a)
; CHECK: fminp s0, v0.2s
        ret <1 x float> %val
}

declare <1 x double> @llvm.aarch64.neon.vpminq(<2 x double>)

define <1 x double> @test_fminp_v1f64(<2 x double> %a) {
; CHECK: test_fminp_v1f64:
        %val = call <1 x double> @llvm.aarch64.neon.vpminq(<2 x double> %a)
; CHECK: fminp d0, v0.2d
        ret <1 x double> %val
}

declare <1 x float> @llvm.aarch64.neon.vpfmaxnm(<2 x float>)

define <1 x float> @test_fmaxnmp_v1f32(<2 x float> %a) {
; CHECK: test_fmaxnmp_v1f32:
        %val = call <1 x float> @llvm.aarch64.neon.vpfmaxnm(<2 x float> %a)
; CHECK: fmaxnmp s0, v0.2s
        ret <1 x float> %val
}

declare <1 x double> @llvm.aarch64.neon.vpfmaxnmq(<2 x double>)

define <1 x double> @test_fmaxnmp_v1f64(<2 x double> %a) {
; CHECK: test_fmaxnmp_v1f64:
        %val = call <1 x double> @llvm.aarch64.neon.vpfmaxnmq(<2 x double> %a)
; CHECK: fmaxnmp d0, v0.2d
        ret <1 x double> %val
}

declare <1 x float> @llvm.aarch64.neon.vpfminnm(<2 x float>)

define <1 x float> @test_fminnmp_v1f32(<2 x float> %a) {
; CHECK: test_fminnmp_v1f32:
        %val = call <1 x float> @llvm.aarch64.neon.vpfminnm(<2 x float> %a)
; CHECK: fminnmp s0, v0.2s
        ret <1 x float> %val
}

declare <1 x double> @llvm.aarch64.neon.vpfminnmq(<2 x double>)

define <1 x double> @test_fminnmp_v1f64(<2 x double> %a) {
; CHECK: test_fminnmp_v1f64:
        %val = call <1 x double> @llvm.aarch64.neon.vpfminnmq(<2 x double> %a)
; CHECK: fminnmp d0, v0.2d
        ret <1 x double> %val
}

define float @test_vaddv_f32(<2 x float> %a) {
; CHECK-LABEL: test_vaddv_f32
; CHECK: faddp {{s[0-9]+}}, {{v[0-9]+}}.2s
  %1 = tail call <1 x float> @llvm.aarch64.neon.vaddv.v1f32.v2f32(<2 x float> %a)
  %2 = extractelement <1 x float> %1, i32 0
  ret float %2
}

define float @test_vaddvq_f32(<4 x float> %a) {
; CHECK-LABEL: test_vaddvq_f32
; CHECK: faddp {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK: faddp {{s[0-9]+}}, {{v[0-9]+}}.2s
  %1 = tail call <1 x float> @llvm.aarch64.neon.vaddv.v1f32.v4f32(<4 x float> %a)
  %2 = extractelement <1 x float> %1, i32 0
  ret float %2
}

define double @test_vaddvq_f64(<2 x double> %a) {
; CHECK-LABEL: test_vaddvq_f64
; CHECK: faddp {{d[0-9]+}}, {{v[0-9]+}}.2d
  %1 = tail call <1 x double> @llvm.aarch64.neon.vaddv.v1f64.v2f64(<2 x double> %a)
  %2 = extractelement <1 x double> %1, i32 0
  ret double %2
}

define float @test_vmaxv_f32(<2 x float> %a) {
; CHECK-LABEL: test_vmaxv_f32
; CHECK: fmaxp {{s[0-9]+}}, {{v[0-9]+}}.2s
  %1 = tail call <1 x float> @llvm.aarch64.neon.vmaxv.v1f32.v2f32(<2 x float> %a)
  %2 = extractelement <1 x float> %1, i32 0
  ret float %2
}

define double @test_vmaxvq_f64(<2 x double> %a) {
; CHECK-LABEL: test_vmaxvq_f64
; CHECK: fmaxp {{d[0-9]+}}, {{v[0-9]+}}.2d
  %1 = tail call <1 x double> @llvm.aarch64.neon.vmaxv.v1f64.v2f64(<2 x double> %a)
  %2 = extractelement <1 x double> %1, i32 0
  ret double %2
}

define float @test_vminv_f32(<2 x float> %a) {
; CHECK-LABEL: test_vminv_f32
; CHECK: fminp {{s[0-9]+}}, {{v[0-9]+}}.2s
  %1 = tail call <1 x float> @llvm.aarch64.neon.vminv.v1f32.v2f32(<2 x float> %a)
  %2 = extractelement <1 x float> %1, i32 0
  ret float %2
}

define double @test_vminvq_f64(<2 x double> %a) {
; CHECK-LABEL: test_vminvq_f64
; CHECK: fminp {{d[0-9]+}}, {{v[0-9]+}}.2d
  %1 = tail call <1 x double> @llvm.aarch64.neon.vminv.v1f64.v2f64(<2 x double> %a)
  %2 = extractelement <1 x double> %1, i32 0
  ret double %2
}

define double @test_vmaxnmvq_f64(<2 x double> %a) {
; CHECK-LABEL: test_vmaxnmvq_f64
; CHECK: fmaxnmp {{d[0-9]+}}, {{v[0-9]+}}.2d
  %1 = tail call <1 x double> @llvm.aarch64.neon.vmaxnmv.v1f64.v2f64(<2 x double> %a)
  %2 = extractelement <1 x double> %1, i32 0
  ret double %2
}

define float @test_vmaxnmv_f32(<2 x float> %a) {
; CHECK-LABEL: test_vmaxnmv_f32
; CHECK: fmaxnmp {{s[0-9]+}}, {{v[0-9]+}}.2s
  %1 = tail call <1 x float> @llvm.aarch64.neon.vmaxnmv.v1f32.v2f32(<2 x float> %a)
  %2 = extractelement <1 x float> %1, i32 0
  ret float %2
}

define double @test_vminnmvq_f64(<2 x double> %a) {
; CHECK-LABEL: test_vminnmvq_f64
; CHECK: fminnmp {{d[0-9]+}}, {{v[0-9]+}}.2d
  %1 = tail call <1 x double> @llvm.aarch64.neon.vminnmv.v1f64.v2f64(<2 x double> %a)
  %2 = extractelement <1 x double> %1, i32 0
  ret double %2
}

define float @test_vminnmv_f32(<2 x float> %a) {
; CHECK-LABEL: test_vminnmv_f32
; CHECK: fminnmp {{s[0-9]+}}, {{v[0-9]+}}.2s
  %1 = tail call <1 x float> @llvm.aarch64.neon.vminnmv.v1f32.v2f32(<2 x float> %a)
  %2 = extractelement <1 x float> %1, i32 0
  ret float %2
}

define <2 x i64> @test_vpaddq_s64(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vpaddq_s64
; CHECK: addp {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
  %1 = tail call <2 x i64> @llvm.arm.neon.vpadd.v2i64(<2 x i64> %a, <2 x i64> %b)
  ret <2 x i64> %1
}

define <2 x i64> @test_vpaddq_u64(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: test_vpaddq_u64
; CHECK: addp {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
  %1 = tail call <2 x i64> @llvm.arm.neon.vpadd.v2i64(<2 x i64> %a, <2 x i64> %b)
  ret <2 x i64> %1
}

define i64 @test_vaddvq_s64(<2 x i64> %a) {
; CHECK-LABEL: test_vaddvq_s64
; CHECK: addp {{d[0-9]+}}, {{v[0-9]+}}.2d
  %1 = tail call <1 x i64> @llvm.aarch64.neon.vaddv.v1i64.v2i64(<2 x i64> %a)
  %2 = extractelement <1 x i64> %1, i32 0
  ret i64 %2
}

define i64 @test_vaddvq_u64(<2 x i64> %a) {
; CHECK-LABEL: test_vaddvq_u64
; CHECK: addp {{d[0-9]+}}, {{v[0-9]+}}.2d
  %1 = tail call <1 x i64> @llvm.aarch64.neon.vaddv.v1i64.v2i64(<2 x i64> %a)
  %2 = extractelement <1 x i64> %1, i32 0
  ret i64 %2
}

declare <1 x i64> @llvm.aarch64.neon.vaddv.v1i64.v2i64(<2 x i64>)

declare <2 x i64> @llvm.arm.neon.vpadd.v2i64(<2 x i64>, <2 x i64>)

declare <1 x float> @llvm.aarch64.neon.vminnmv.v1f32.v2f32(<2 x float>)

declare <1 x double> @llvm.aarch64.neon.vminnmv.v1f64.v2f64(<2 x double>)

declare <1 x float> @llvm.aarch64.neon.vmaxnmv.v1f32.v2f32(<2 x float>)

declare <1 x double> @llvm.aarch64.neon.vmaxnmv.v1f64.v2f64(<2 x double>)

declare <1 x double> @llvm.aarch64.neon.vminv.v1f64.v2f64(<2 x double>)

declare <1 x float> @llvm.aarch64.neon.vminv.v1f32.v2f32(<2 x float>)

declare <1 x double> @llvm.aarch64.neon.vmaxv.v1f64.v2f64(<2 x double>)

declare <1 x float> @llvm.aarch64.neon.vmaxv.v1f32.v2f32(<2 x float>)

declare <1 x double> @llvm.aarch64.neon.vaddv.v1f64.v2f64(<2 x double>)

declare <1 x float> @llvm.aarch64.neon.vaddv.v1f32.v4f32(<4 x float>)

declare <1 x float> @llvm.aarch64.neon.vaddv.v1f32.v2f32(<2 x float>)