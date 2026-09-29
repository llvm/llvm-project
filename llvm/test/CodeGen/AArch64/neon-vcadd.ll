; RUN: llc %s -mtriple=aarch64 -mattr=+v8.3a,+fullfp16 -o - | FileCheck %s

define <4 x half> @foo16x4_rot(<4 x half> %a, <4 x half> %b) {
entry:
; CHECK-LABEL: foo16x4_rot
; CHECK-DAG: fcadd v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, #90
; CHECK-DAG: fcadd v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, #270
  %vcadd_rot90_v2.i = tail call <4 x half> @llvm.aarch64.neon.vcadd.rot90.v4f16(<4 x half> %a, <4 x half> %b)
  %vcadd_rot270_v2.i = tail call <4 x half> @llvm.aarch64.neon.vcadd.rot270.v4f16(<4 x half> %a, <4 x half> %b)
  %add = fadd <4 x half> %vcadd_rot90_v2.i, %vcadd_rot270_v2.i
  ret <4 x half> %add
}

define <2 x float> @foo32x2_rot(<2 x float> %a, <2 x float> %b) {
entry:
; CHECK-LABEL: foo32x2_rot
; CHECK-DAG: fcadd v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, #90
; CHECK-DAG: fcadd v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, #270
  %vcadd_rot90_v2.i = tail call <2 x float> @llvm.aarch64.neon.vcadd.rot90.v2f32(<2 x float> %a, <2 x float> %b)
  %vcadd_rot270_v2.i = tail call <2 x float> @llvm.aarch64.neon.vcadd.rot270.v2f32(<2 x float> %a, <2 x float> %b)
  %add = fadd <2 x float> %vcadd_rot90_v2.i, %vcadd_rot270_v2.i
  ret <2 x float> %add
}

define <8 x half> @foo16x8_rot(<8 x half> %a, <8 x half> %b) {
entry:
; CHECK-LABEL: foo16x8_rot
; CHECK-DAG: fcadd v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, #90
; CHECK-DAG: fcadd v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, #270
  %vcaddq_rot90_v2.i = tail call <8 x half> @llvm.aarch64.neon.vcadd.rot90.v8f16(<8 x half> %a, <8 x half> %b)
  %vcaddq_rot270_v2.i = tail call <8 x half> @llvm.aarch64.neon.vcadd.rot270.v8f16(<8 x half> %a, <8 x half> %b)
  %add = fadd <8 x half> %vcaddq_rot90_v2.i, %vcaddq_rot270_v2.i
  ret <8 x half> %add
}

define <4 x float> @foo32x4_rot(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: foo32x4_rot
; CHECK-DAG: fcadd v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, #90
; CHECK-DAG: fcadd v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, #270
  %vcaddq_rot90_v2.i = tail call <4 x float> @llvm.aarch64.neon.vcadd.rot90.v4f32(<4 x float> %a, <4 x float> %b)
  %vcaddq_rot270_v2.i = tail call <4 x float> @llvm.aarch64.neon.vcadd.rot270.v4f32(<4 x float> %a, <4 x float> %b)
  %add = fadd <4 x float> %vcaddq_rot90_v2.i, %vcaddq_rot270_v2.i
  ret <4 x float> %add
}

define <2 x double> @foo64x2_rot(<2 x double> %a, <2 x double> %b) {
entry:
; CHECK-LABEL: foo64x2_rot
; CHECK-DAG: fcadd v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, #90
; CHECK-DAG: fcadd v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, #270
  %vcaddq_rot90_v2.i = tail call <2 x double> @llvm.aarch64.neon.vcadd.rot90.v2f64(<2 x double> %a, <2 x double> %b)
  %vcaddq_rot270_v2.i = tail call <2 x double> @llvm.aarch64.neon.vcadd.rot270.v2f64(<2 x double> %a, <2 x double> %b)
  %add = fadd <2 x double> %vcaddq_rot90_v2.i, %vcaddq_rot270_v2.i
  ret <2 x double> %add
}

declare <4 x half> @llvm.aarch64.neon.vcadd.rot90.v4f16(<4 x half>, <4 x half>)
declare <4 x half> @llvm.aarch64.neon.vcadd.rot270.v4f16(<4 x half>, <4 x half>)
declare <2 x float> @llvm.aarch64.neon.vcadd.rot90.v2f32(<2 x float>, <2 x float>)
declare <2 x float> @llvm.aarch64.neon.vcadd.rot270.v2f32(<2 x float>, <2 x float>)
declare <8 x half> @llvm.aarch64.neon.vcadd.rot90.v8f16(<8 x half>, <8 x half>)
declare <8 x half> @llvm.aarch64.neon.vcadd.rot270.v8f16(<8 x half>, <8 x half>)
declare <4 x float> @llvm.aarch64.neon.vcadd.rot90.v4f32(<4 x float>, <4 x float>)
declare <4 x float> @llvm.aarch64.neon.vcadd.rot270.v4f32(<4 x float>, <4 x float>)
declare <2 x double> @llvm.aarch64.neon.vcadd.rot90.v2f64(<2 x double>, <2 x double>)
declare <2 x double> @llvm.aarch64.neon.vcadd.rot270.v2f64(<2 x double>, <2 x double>)
