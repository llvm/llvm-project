; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

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
; CHECK-SPIRV:       %[[#]] = OpDot %[[#]] %[[#]] %[[#]]
; CHECK-SPIRV:       OpFunctionEnd

define spir_kernel void @testVector(<2 x float> %f) {
entry:
  %call = tail call spir_func float @_Z3dotDv2_fS_(<2 x float> %f, <2 x float> %f)
  ret void
}

declare spir_func float @_Z3dotff(float, float)

declare spir_func float @_Z3dotDv2_fS_(<2 x float>, <2 x float>)
