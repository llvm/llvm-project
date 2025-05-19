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
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[DEGREES:%.*]] = fmul float [[A]], 0x404CA5DC20000000
; CHECK-NEXT:    ret float [[DEGREES]]
;
entry:
  %dx.degrees = call float @llvm.dx.degrees.f32(float %a)
  ret float %dx.degrees
}

define noundef <4 x float> @degrees_float4(<4 x float> noundef %a) #0 {
; CHECK-LABEL: define noundef <4 x float> @degrees_float4(
; CHECK-SAME: <4 x float> noundef [[A:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A0:%.*]] = extractelement <4 x float> [[A]], i64 0
; CHECK-NEXT:    [[DEGREES_A0:%.*]] = fmul float [[A0]], 0x404CA5DC20000000
; CHECK-NEXT:    [[A1:%.*]] = extractelement <4 x float> [[A]], i64 1
; CHECK-NEXT:    [[DEGREES_A1:%.*]] = fmul float [[A1]], 0x404CA5DC20000000
; CHECK-NEXT:    [[A2:%.*]] = extractelement <4 x float> [[A]], i64 2
; CHECK-NEXT:    [[DEGREES_A2:%.*]] = fmul float [[A2]], 0x404CA5DC20000000
; CHECK-NEXT:    [[A3:%.*]] = extractelement <4 x float> [[A]], i64 3
; CHECK-NEXT:    [[DEGREES_A3:%.*]] = fmul float [[A3]], 0x404CA5DC20000000
; CHECK-NEXT:    [[INSERT_0:%.*]] = insertelement <4 x float> poison, float [[DEGREES_A0]], i64 0
; CHECK-NEXT:    [[INSERT_1:%.*]] = insertelement <4 x float> [[INSERT_0]], float [[DEGREES_A1]], i64 1
; CHECK-NEXT:    [[INSERT_2:%.*]] = insertelement <4 x float> [[INSERT_1]], float [[DEGREES_A2]], i64 2
; CHECK-NEXT:    [[RES:%.*]] = insertelement <4 x float> [[INSERT_2]], float [[DEGREES_A3]], i64 3
; CHECK-NEXT:    ret <4 x float> [[RES]]
;
entry:
  %2 = call <4 x float> @llvm.dx.degrees.v4f32(<4 x float> %a)
  ret <4 x float> %2
}

declare half  @llvm.dx.degrees.f16(half)
declare float @llvm.dx.degrees.f32(float)
declare <4 x float> @llvm.dx.degrees.v4f32(<4 x float>)
