; RUN: llc < %s -march=nvptx64 -mcpu=sm_80 -mattr=+ptx70 | FileCheck %s
; RUN: %if ptxas-11.0 %{ llc < %s -march=nvptx64 -mcpu=sm_80 -mattr=+ptx70 | %ptxas-verify -arch=sm_80 %}


; CHECK-LABEL: cvt_rn_bf16x2_f32
define <2 x bfloat> @cvt_rn_bf16x2_f32(float %f1, float %f2) {

; CHECK: cvt.rn.bf16x2.f32
  %val = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rn(float %f1, float %f2);

ret <2 x bfloat> %val
}

; CHECK-LABEL: cvt_rn_relu_bf16x2_f32
define <2 x bfloat> @cvt_rn_relu_bf16x2_f32(float %f1, float %f2) {

; CHECK: cvt.rn.relu.bf16x2.f32
%val = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rn.relu(float %f1, float %f2);

ret <2 x bfloat> %val
}

; CHECK-LABEL: cvt_rz_bf16x2_f32
define <2 x bfloat> @cvt_rz_bf16x2_f32(float %f1, float %f2) {

; CHECK: cvt.rz.bf16x2.f32
  %val = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rz(float %f1, float %f2);

ret <2 x bfloat> %val
}

; CHECK-LABEL: cvt_rz_relu_bf16x2_f32
define <2 x bfloat> @cvt_rz_relu_bf16x2_f32(float %f1, float %f2) {

; CHECK: cvt.rz.relu.bf16x2.f32
%val = call <2 x bfloat> @llvm.nvvm.ff2bf16x2.rz.relu(float %f1, float %f2);

ret <2 x bfloat> %val
}

declare <2 x bfloat> @llvm.nvvm.ff2bf16x2.rn(float, float)
declare <2 x bfloat> @llvm.nvvm.ff2bf16x2.rn.relu(float, float)
declare <2 x bfloat> @llvm.nvvm.ff2bf16x2.rz(float, float)
declare <2 x bfloat> @llvm.nvvm.ff2bf16x2.rz.relu(float, float)

; CHECK-LABEL: cvt_rn_f16x2_f32
define <2 x half> @cvt_rn_f16x2_f32(float %f1, float %f2) {

; CHECK: cvt.rn.f16x2.f32
  %val = call <2 x half> @llvm.nvvm.ff2f16x2.rn(float %f1, float %f2);

ret <2 x half> %val
}

; CHECK-LABEL: cvt_rn_relu_f16x2_f32
define <2 x half> @cvt_rn_relu_f16x2_f32(float %f1, float %f2) {

; CHECK: cvt.rn.relu.f16x2.f32
%val = call <2 x half> @llvm.nvvm.ff2f16x2.rn.relu(float %f1, float %f2);

ret <2 x half> %val
}

; CHECK-LABEL: cvt_rz_f16x2_f32
define <2 x half> @cvt_rz_f16x2_f32(float %f1, float %f2) {

; CHECK: cvt.rz.f16x2.f32
  %val = call <2 x half> @llvm.nvvm.ff2f16x2.rz(float %f1, float %f2);

ret <2 x half> %val
}

; CHECK-LABEL: cvt_rz_relu_f16x2_f32
define <2 x half> @cvt_rz_relu_f16x2_f32(float %f1, float %f2) {

; CHECK: cvt.rz.relu.f16x2.f32
%val = call <2 x half> @llvm.nvvm.ff2f16x2.rz.relu(float %f1, float %f2);

ret <2 x half> %val
}

declare <2 x half> @llvm.nvvm.ff2f16x2.rn(float, float)
declare <2 x half> @llvm.nvvm.ff2f16x2.rn.relu(float, float)
declare <2 x half> @llvm.nvvm.ff2f16x2.rz(float, float)
declare <2 x half> @llvm.nvvm.ff2f16x2.rz.relu(float, float)

; CHECK-LABEL: cvt_rn_bf16_f32
define bfloat @cvt_rn_bf16_f32(float %f1) {

; CHECK: cvt.rn.bf16.f32
  %val = call bfloat @llvm.nvvm.f2bf16.rn(float %f1);

ret bfloat %val
}

; CHECK-LABEL: cvt_rn_relu_bf16_f32
define bfloat @cvt_rn_relu_bf16_f32(float %f1) {

; CHECK: cvt.rn.relu.bf16.f32
%val = call bfloat @llvm.nvvm.f2bf16.rn.relu(float %f1);

ret bfloat %val
}

; CHECK-LABEL: cvt_rz_bf16_f32
define bfloat @cvt_rz_bf16_f32(float %f1) {

; CHECK: cvt.rz.bf16.f32
  %val = call bfloat @llvm.nvvm.f2bf16.rz(float %f1);

ret bfloat %val
}

; CHECK-LABEL: cvt_rz_relu_bf16_f32
define bfloat @cvt_rz_relu_bf16_f32(float %f1) {

; CHECK: cvt.rz.relu.bf16.f32
%val = call bfloat @llvm.nvvm.f2bf16.rz.relu(float %f1);

ret bfloat %val
}

declare bfloat @llvm.nvvm.f2bf16.rn(float)
declare bfloat @llvm.nvvm.f2bf16.rn.relu(float)
declare bfloat @llvm.nvvm.f2bf16.rz(float)
declare bfloat @llvm.nvvm.f2bf16.rz.relu(float)

; CHECK-LABEL: cvt_rna_tf32_f32
define i32 @cvt_rna_tf32_f32(float %f1) {

; CHECK: cvt.rna.tf32.f32
  %val = call i32 @llvm.nvvm.f2tf32.rna(float %f1);

ret i32 %val
}

declare i32 @llvm.nvvm.f2tf32.rna(float)
