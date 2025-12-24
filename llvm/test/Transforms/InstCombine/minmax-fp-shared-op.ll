; RUN: opt -passes=instcombine -S %s -o - | FileCheck %s

declare float @llvm.minnum.f32(float, float)
declare float @llvm.fma.f32(float, float, float)

; CHECK-LABEL: @minnum_shared_op_mixed(
; CHECK: call float @llvm.minnum.f32
; CHECK: call float @llvm.fma.f32
; CHECK: call float @llvm.minnum.f32
define float @minnum_shared_op_mixed(float %x) {
entry:
  %m0 = call float @llvm.minnum.f32(float %x, float 0.000000e+00)
  %f = call float @llvm.fma.f32(float %x, float 0.000000e+00, float %x)
  %m2 = call float @llvm.minnum.f32(float %f, float %m0)
  ret float %m2
}
