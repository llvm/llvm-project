; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_KHR_bfloat16 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_KHR_bfloat16 %s -o - -filetype=obj | spirv-val %}
; XFAIL: *

; CHECK-SPIRV-DAG: OpCapability CooperativeMatrixKHR
; CHECK-SPIRV-DAG: OpCapability BFloat16TypeKHR
; CHECK-SPIRV-DAG: OpCapability BFloat16CooperativeMatrixKHR
; CHECK-SPIRV-DAG: OpExtension "SPV_KHR_cooperative_matrix"
; CHECK-SPIRV-DAG: OpExtension "SPV_KHR_bfloat16"

; CHECK-SPIRV-DAG: %[[#BFloatTy:]] = OpTypeFloat 16 0
; CHECK-SPIRV-DAG: %[[#Int32Ty:]]= OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#Const12:]] = OpConstant %[[#Int32Ty]] 12
; CHECK-SPIRV-DAG: %[[#Const3:]] = OpConstant %[[#Int32Ty]] 3
; CHECK-SPIRV-DAG: %[[#Const2:]] = OpConstant %[[#Int32Ty]] 2
; CHECK-SPIRV-DAG: %[[#MatTy:]] = OpTypeCooperativeMatrixKHR %[[#BFloatTy]] %[[#Const3]] %[[#Const12]] %[[#Const12]] %[[#Const2]]
; CHECK-SPIRV-DAG: %[[#]] = OpConstant %[[#BFloatTy]] 16256
; CHECK-SPIRV: %[[#]] = OpCompositeConstruct %[[#MatTy]]

declare spir_func target("spirv.CooperativeMatrixKHR", bfloat, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructu6__bf16(bfloat)

define spir_kernel void @test() {
  %mat = call spir_func target("spirv.CooperativeMatrixKHR", bfloat, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructu6__bf16(bfloat 1.0)
  ret void
}
