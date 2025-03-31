
; RUN: opt -S -dxil-legalize -mtriple=dxil-pc-shadermodel6.3-library %s -o - | FileCheck %s

define noundef half @frem_half(half noundef %a, half noundef %b) {
; CHECK-LABEL: define noundef half @frem_half(
; CHECK-SAME: half noundef [[A:%.*]], half noundef [[B:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[FDIV:%.*]] = fdiv half [[A]], [[B]]
; CHECK-NEXT:    [[FCMP:%.*]] = fcmp oge half [[FDIV]], 0xH0000
; CHECK-NEXT:    [[FABS:%.*]] = call half @llvm.fabs.f16(half [[FDIV]])
; CHECK-NEXT:    [[FRAC:%.*]] = call half @llvm.dx.frac.f16(half [[FABS]])
; CHECK-NEXT:    [[FNEG:%.*]] = fneg half [[FRAC]]
; CHECK-NEXT:    [[SELC:%.*]] = select i1 [[FCMP]], half [[FRAC]], half [[FNEG]]
; CHECK-NEXT:    [[FMUL:%.*]] = fmul half [[SELC]], [[B]]
; CHECK-NEXT:    ret half [[FMUL]]
;
entry:
  %fmod.i = frem reassoc nnan ninf nsz arcp afn half %a, %b
  ret half %fmod.i
}

; Note by the time the legalizer sees frem with vec type frem will be scalarized
; This test is for completeness not for expected input of DXL SMs <= 6.8.

define noundef <2 x half> @frem_half2(<2 x half> noundef %a, <2 x half> noundef %b) {
; CHECK-LABEL: define noundef <2 x half> @frem_half2(
; CHECK-SAME: <2 x half> noundef [[A:%.*]], <2 x half> noundef [[B:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[FDIV:%.*]] = fdiv <2 x half> [[A]], [[B]]
; CHECK-NEXT:    [[FCMP:%.*]] = fcmp oge <2 x half> [[FDIV]], zeroinitializer
; CHECK-NEXT:    [[FABS:%.*]] = call <2 x half> @llvm.fabs.v2f16(<2 x half> [[FDIV]])
; CHECK-NEXT:    [[FRAC:%.*]] = call <2 x half> @llvm.dx.frac.v2f16(<2 x half> [[FABS]])
; CHECK-NEXT:    [[FNEG:%.*]] = fneg <2 x half> [[FRAC]]
; CHECK-NEXT:    [[SELC:%.*]] = select  <2 x i1> [[FCMP]], <2 x half> [[FRAC]], <2 x half> [[FNEG]]
; CHECK-NEXT:    [[FMUL:%.*]] = fmul <2 x half> [[SELC]], [[B]]
; CHECK-NEXT:    ret <2 x half> [[FMUL]]
;
entry:
  %fmod.i = frem reassoc nnan ninf nsz arcp afn <2 x half> %a, %b
  ret <2 x half> %fmod.i
}

define noundef float @frem_float(float noundef %a, float noundef %b) {
; CHECK-LABEL: define noundef float @frem_float(
; CHECK-SAME: float noundef [[A:%.*]], float noundef [[B:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[FDIV:%.*]] = fdiv float [[A]], [[B]]
; CHECK-NEXT:    [[FCMP:%.*]] = fcmp oge float [[FDIV]], 0.000000e+00 
; CHECK-NEXT:    [[FABS:%.*]] = call float @llvm.fabs.f32(float [[FDIV]])
; CHECK-NEXT:    [[FRAC:%.*]] = call float @llvm.dx.frac.f32(float [[FABS]])
; CHECK-NEXT:    [[FNEG:%.*]] = fneg float [[FRAC]]
; CHECK-NEXT:    [[SELC:%.*]] = select i1 [[FCMP]], float [[FRAC]], float [[FNEG]]
; CHECK-NEXT:    [[FMUL:%.*]] = fmul float [[SELC]], [[B]]
; CHECK-NEXT:    ret float [[FMUL]]
;
entry:
  %fmod.i = frem reassoc nnan ninf nsz arcp afn float %a, %b
  ret float %fmod.i
}
