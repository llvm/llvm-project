; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_bfloat16,+SPV_KHR_cooperative_matrix %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_bfloat16,+SPV_KHR_cooperative_matrix %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability BFloat16TypeKHR
; CHECK-DAG: OpCapability CooperativeMatrixKHR
; CHECK-DAG: OpCapability BFloat16CooperativeMatrixKHR
; CHECK-DAG: OpExtension "SPV_KHR_bfloat16"
; CHECK-DAG: OpExtension "SPV_KHR_cooperative_matrix"
; CHECK: %[[#BFLOAT:]] = OpTypeFloat 16 0
; CHECK: %[[#MatTy:]] = OpTypeCooperativeMatrixKHR %[[#BFLOAT]]  %[[#]] %[[#]] %[[#]] %[[#]]
; CHECK: OpCompositeConstruct %[[#MatTy]] %[[#]]

define spir_kernel void @matr_mult(ptr addrspace(1) align 1 %_arg_accA, ptr addrspace(1) align 1 %_arg_accB, ptr addrspace(1) align 4 %_arg_accC, i64 %_arg_N, i64 %_arg_K) {
entry:
    %addr1 = alloca target("spirv.CooperativeMatrixKHR", bfloat, 3, 12, 12, 2), align 4
    %res = alloca target("spirv.CooperativeMatrixKHR", bfloat, 3, 12, 12, 2), align 4
    %m1 = tail call spir_func target("spirv.CooperativeMatrixKHR", bfloat, 3, 12, 12, 2) @_Z26__spirv_CompositeConstruct(bfloat 1.0)
    store target("spirv.CooperativeMatrixKHR", bfloat, 3, 12, 12, 2) %m1, ptr %addr1, align 4
    ret void
}

declare dso_local spir_func target("spirv.CooperativeMatrixKHR", bfloat, 3, 12, 12, 2) @_Z26__spirv_CompositeConstruct(bfloat)
