; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_89 -mattr=+ptx81 | FileCheck %s
; RUN: %if ptxas-12.1 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_89 -mattr=+ptx81 | %ptxas-verify -arch=sm_89 %}

; CHECK-LABEL: cvt_rn_e4m3x2_f32
define i16 @cvt_rn_e4m3x2_f32(float %f1, float %f2) {
; CHECK: cvt.rn.satfinite.e4m3x2.f32
  %val = call i16 @llvm.nvvm.ff.to.e4m3x2.rn(float %f1, float %f2);
  ret i16 %val
}

; CHECK-LABEL: cvt_rn_relu_e4m3x2_f32
define i16 @cvt_rn_relu_e4m3x2_f32(float %f1, float %f2) {
; CHECK: cvt.rn.satfinite.relu.e4m3x2.f32
  %val = call i16 @llvm.nvvm.ff.to.e4m3x2.rn.relu(float %f1, float %f2);
  ret i16 %val
}

; CHECK-LABEL: cvt_rn_e5m2x2_f32
define i16 @cvt_rn_e5m2x2_f32(float %f1, float %f2) {
; CHECK: cvt.rn.satfinite.e5m2x2.f32
  %val = call i16 @llvm.nvvm.ff.to.e5m2x2.rn(float %f1, float %f2);
  ret i16 %val
}

; CHECK-LABEL: cvt_rn_relu_e5m2x2_f32
define i16 @cvt_rn_relu_e5m2x2_f32(float %f1, float %f2) {
; CHECK: cvt.rn.satfinite.relu.e5m2x2.f32
  %val = call i16 @llvm.nvvm.ff.to.e5m2x2.rn.relu(float %f1, float %f2);
  ret i16 %val
}

; CHECK-LABEL: cvt_rn_e4m3x2_f16x2
define i16 @cvt_rn_e4m3x2_f16x2(<2 x half> %in) {
; CHECK: cvt.rn.satfinite.e4m3x2.f16x2
  %val = call i16 @llvm.nvvm.f16x2.to.e4m3x2.rn(<2 x half> %in);
  ret i16 %val
}

; CHECK-LABEL: cvt_rn_relu_e4m3x2_f16x2
define i16 @cvt_rn_relu_e4m3x2_f16x2(<2 x half> %in) {
; CHECK: cvt.rn.satfinite.relu.e4m3x2.f16x2
  %val = call i16 @llvm.nvvm.f16x2.to.e4m3x2.rn.relu(<2 x half> %in);
  ret i16 %val
}

; CHECK-LABEL: cvt_rn_e5m2x2_f16x2
define i16 @cvt_rn_e5m2x2_f16x2(<2 x half> %in) {
; CHECK: cvt.rn.satfinite.e5m2x2.f16x2
  %val = call i16 @llvm.nvvm.f16x2.to.e5m2x2.rn(<2 x half> %in);
  ret i16 %val
}

; CHECK-LABEL: cvt_rn_relu_e5m2x2_f16x2
define i16 @cvt_rn_relu_e5m2x2_f16x2(<2 x half> %in) {
; CHECK: cvt.rn.satfinite.relu.e5m2x2.f16x2
  %val = call i16 @llvm.nvvm.f16x2.to.e5m2x2.rn.relu(<2 x half> %in);
  ret i16 %val
}

; CHECK-LABEL: cvt_rn_f16x2_e4m3x2
define <2 x half> @cvt_rn_f16x2_e4m3x2(i16 %in) {
; CHECK: cvt.rn.f16x2.e4m3x2
  %val = call <2 x half> @llvm.nvvm.e4m3x2.to.f16x2.rn(i16 %in);
  ret <2 x half> %val
}

; CHECK-LABEL: cvt_rn_relu_f16x2_e4m3x2
define <2 x half> @cvt_rn_relu_f16x2_e4m3x2(i16 %in) {
; CHECK: cvt.rn.relu.f16x2.e4m3x2
  %val = call <2 x half> @llvm.nvvm.e4m3x2.to.f16x2.rn.relu(i16 %in);
  ret <2 x half> %val
}

; CHECK-LABEL: cvt_rn_f16x2_e5m2x2
define <2 x half> @cvt_rn_f16x2_e5m2x2(i16 %in) {
; CHECK: cvt.rn.f16x2.e5m2x2
  %val = call <2 x half> @llvm.nvvm.e5m2x2.to.f16x2.rn(i16 %in);
  ret <2 x half> %val
}

; CHECK-LABEL: cvt_rn_relu_f16x2_e5m2x2
define <2 x half> @cvt_rn_relu_f16x2_e5m2x2(i16 %in) {
; CHECK: cvt.rn.relu.f16x2.e5m2x2
  %val = call <2 x half> @llvm.nvvm.e5m2x2.to.f16x2.rn.relu(i16 %in);
  ret <2 x half> %val
}

; CHECK-LABEL: cvt_rna_satfinite_tf32_f32
define i32 @cvt_rna_satfinite_tf32_f32(float %f1) {
; CHECK: cvt.rna.satfinite.tf32.f32
  %val = call i32 @llvm.nvvm.f2tf32.rna.satfinite(float %f1)
  ret i32 %val
}
