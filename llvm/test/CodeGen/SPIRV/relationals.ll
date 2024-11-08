; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

declare dso_local spir_func <4 x i8> @_Z13__spirv_IsNanIDv4_aDv4_fET_T0_(<4 x float>)
declare dso_local spir_func <4 x i8> @_Z13__spirv_IsInfIDv4_aDv4_fET_T0_(<4 x float>)
declare dso_local spir_func <4 x i8> @_Z16__spirv_IsFiniteIDv4_aDv4_fET_T0_(<4 x float>)
declare dso_local spir_func <4 x i8> @_Z16__spirv_IsNormalIDv4_aDv4_fET_T0_(<4 x float>)
declare dso_local spir_func <4 x i8> @_Z18__spirv_SignBitSetIDv4_aDv4_fET_T0_(<4 x float>)

; CHECK-SPIRV: %[[#TBool:]] = OpTypeBool
; CHECK-SPIRV: %[[#TBoolVec:]] = OpTypeVector %[[#TBool]]

define spir_kernel void @k() {
entry:
  %arg1 = alloca <4 x float>, align 16
  %ret = alloca <4 x i8>, align 4
  %0 = load <4 x float>, <4 x float>* %arg1, align 16
  %call1 = call spir_func <4 x i8> @_Z13__spirv_IsNanIDv4_aDv4_fET_T0_(<4 x float> %0)
; CHECK-SPIRV: %[[#IsNanRes:]] = OpIsNan %[[#TBoolVec]]
; CHECK-SPIRV: %[[#SelectRes:]] = OpSelect %[[#]] %[[#IsNanRes]]
; CHECK-SPIRV: OpStore %[[#]] %[[#SelectRes]]
  store <4 x i8> %call1, <4 x i8>* %ret, align 4
  %call2 = call spir_func <4 x i8> @_Z13__spirv_IsInfIDv4_aDv4_fET_T0_(<4 x float> %0)
; CHECK-SPIRV: %[[#IsInfRes:]] = OpIsInf %[[#TBoolVec]]
; CHECK-SPIRV: %[[#Select1Res:]] = OpSelect %[[#]] %[[#IsInfRes]]
; CHECK-SPIRV: OpStore %[[#]] %[[#Select1Res]]
  store <4 x i8> %call2, <4 x i8>* %ret, align 4
  %call3 = call spir_func <4 x i8> @_Z16__spirv_IsFiniteIDv4_aDv4_fET_T0_(<4 x float> %0)
; CHECK-SPIRV: %[[#IsFiniteRes:]] = OpIsFinite %[[#TBoolVec]]
; CHECK-SPIRV: %[[#Select2Res:]] = OpSelect %[[#]] %[[#IsFiniteRes]]
; CHECK-SPIRV: OpStore %[[#]] %[[#Select2Res]]
  store <4 x i8> %call3, <4 x i8>* %ret, align 4
  %call4 = call spir_func <4 x i8> @_Z16__spirv_IsNormalIDv4_aDv4_fET_T0_(<4 x float> %0)
; CHECK-SPIRV: %[[#IsNormalRes:]] = OpIsNormal %[[#TBoolVec]]
; CHECK-SPIRV: %[[#Select3Res:]] = OpSelect %[[#]] %[[#IsNormalRes]]
; CHECK-SPIRV: OpStore %[[#]] %[[#Select3Res]]
  store <4 x i8> %call4, <4 x i8>* %ret, align 4
  %call5 = call spir_func <4 x i8> @_Z18__spirv_SignBitSetIDv4_aDv4_fET_T0_(<4 x float> %0)
; CHECK-SPIRV: %[[#SignBitSetRes:]] = OpSignBitSet %[[#TBoolVec]]
; CHECK-SPIRV: %[[#Select4Res:]] = OpSelect %[[#]] %[[#SignBitSetRes]]
; CHECK-SPIRV: OpStore %[[#]] %[[#Select4Res]]
  store <4 x i8> %call5, <4 x i8>* %ret, align 4
  ret void
}
