; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: %[[#TyFloat:]] = OpTypeFloat 32
; CHECK-SPIRV-DAG: %[[#TyHalf:]] = OpTypeFloat 16

;; The OpDot operands must be vectors; check that translating dot with
;; scalar arguments does not result in OpDot.
; CHECK-SPIRV-LABEL: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-SPIRV:       %[[#]] = OpFMul %[[#]] %[[#]] %[[#]]
; CHECK-SPIRV-NOT:   %[[#]] = OpDot %[[#]] %[[#]] %[[#]]
; CHECK-SPIRV:       OpFunctionEnd

define spir_kernel void @testScalar(float %f) {
entry:
  %call = tail call spir_func float @_Z3dotff(float %f, float %f)
  ret void
}

;; The OpDot operands must be vectors; check that translating dot with
;; vector arguments results in OpDot.
; CHECK-SPIRV-LABEL: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-SPIRV:       %[[#]] = OpDot %[[#TyFloat]] %[[#]] %[[#]]
; CHECK-SPIRV:       %[[#]] = OpDot %[[#TyFloat]] %[[#]] %[[#]]
; CHECK-SPIRV:       %[[#]] = OpDot %[[#TyHalf]] %[[#]] %[[#]]
; CHECK-SPIRV:       OpFunctionEnd

define spir_kernel void @testVector(<2 x float> %f, <2 x half> %h) {
entry:
  %call = tail call spir_func float @_Z3dotDv2_fS_(<2 x float> %f, <2 x float> %f)
  %call2 = tail call spir_func float @__spirv_Dot(<2 x float> %f, <2 x float> %f)
  %call3 = tail call spir_func half @_Z11__spirv_DotDv2_DF16_S_(<2 x half> %h, <2 x half> %h)
  ret void
}

declare spir_func float @_Z3dotff(float, float)

declare spir_func float @_Z3dotDv2_fS_(<2 x float>, <2 x float>)
declare spir_func float @__spirv_Dot(<2 x float>, <2 x float>)
declare spir_func half @_Z11__spirv_DotDv2_DF16_S_(<2 x half>, <2 x half>)
