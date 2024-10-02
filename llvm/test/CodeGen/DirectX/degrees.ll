; RUN: opt -S -dxil-intrinsic-expansion -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil op function calls for degrees are expanded and lowered as fmul for float and half.

define noundef half @degrees_half(half noundef %a) {
; CHECK-LABEL: define noundef half @degrees_half(
; CHECK-SAME: half noundef [[A:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[DX_DEGREES1:%.*]] = fmul half [[A]], 0xH5329
; CHECK-NEXT:    ret half [[DX_DEGREES1]]
;
entry:
  %dx.degrees = call half @llvm.dx.degrees.f16(half %a)
  ret half %dx.degrees
}

define noundef float @degrees_float(float noundef %a) #0 {
; CHECK-LABEL: define noundef float @degrees_float(
; CHECK-SAME: float noundef [[A:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[DX_DEGREES1:%.*]] = fmul float [[A]], 0x404CA5DC20000000
; CHECK-NEXT:    ret float [[DX_DEGREES1]]
;
entry:
  %dx.degrees = call float @llvm.dx.degrees.f32(float %a)
  ret float %dx.degrees
}

define noundef <4 x float> @degrees_float4(<4 x float> noundef %a) #0 {
; CHECK-LABEL: define noundef <4 x float> @degrees_float4(
; CHECK-SAME: <4 x float> noundef [[A:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[A_I0:%.*]] = extractelement <4 x float> [[A]], i64 0
; CHECK-NEXT:    [[DOTI04:%.*]] = fmul float [[A_I0]], 0x404CA5DC20000000
; CHECK-NEXT:    [[A_I1:%.*]] = extractelement <4 x float> [[A]], i64 1
; CHECK-NEXT:    [[DOTI13:%.*]] = fmul float [[A_I1]], 0x404CA5DC20000000
; CHECK-NEXT:    [[A_I2:%.*]] = extractelement <4 x float> [[A]], i64 2
; CHECK-NEXT:    [[DOTI22:%.*]] = fmul float [[A_I2]], 0x404CA5DC20000000
; CHECK-NEXT:    [[A_I3:%.*]] = extractelement <4 x float> [[A]], i64 3
; CHECK-NEXT:    [[DOTI31:%.*]] = fmul float [[A_I3]], 0x404CA5DC20000000
; CHECK-NEXT:    [[DOTUPTO0:%.*]] = insertelement <4 x float> poison, float [[DOTI04]], i64 0
; CHECK-NEXT:    [[DOTUPTO1:%.*]] = insertelement <4 x float> [[DOTUPTO0]], float [[DOTI13]], i64 1
; CHECK-NEXT:    [[DOTUPTO2:%.*]] = insertelement <4 x float> [[DOTUPTO1]], float [[DOTI22]], i64 2
; CHECK-NEXT:    [[TMP0:%.*]] = insertelement <4 x float> [[DOTUPTO2]], float [[DOTI31]], i64 3
; CHECK-NEXT:    ret <4 x float> [[TMP0]]
;
entry:
  %2 = call <4 x float> @llvm.dx.degrees.v4f32(<4 x float> %a)
  ret <4 x float> %2
}

declare half  @llvm.dx.degrees.f16(half)
declare float @llvm.dx.degrees.f32(float)
declare <4 x float> @llvm.dx.degrees.v4f32(<4 x float>)
