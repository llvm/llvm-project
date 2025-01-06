; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_cooperative_matrix %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: LLVM ERROR: MatrixAAndBTF32ComponentsINTEL type interpretation require the following SPIR-V extension: SPV_INTEL_joint_matrix

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_INTEL_joint_matrix %s -o - | FileCheck %s

; CHECK-DAG: Capability CooperativeMatrixKHR
; CHECK-DAG: Extension "SPV_KHR_cooperative_matrix"
; CHECK-DAG: Extension "SPV_INTEL_joint_matrix"
; CHECK-DAG: OpCapability CooperativeMatrixTF32ComponentTypeINTEL

; CHECK: OpCooperativeMatrixMulAddKHR %[[#]] %[[#]] %[[#]] %[[#]] MatrixAAndBTF32ComponentsINTEL

define weak_odr dso_local spir_kernel void @_ZTSZZ15matrix_multiply(ptr addrspace(1) noundef align 1 %_arg_accA, ptr addrspace(1) noundef align 1 %_arg_accB, ptr addrspace(1) noundef align 1 %_arg_accC, i64 noundef %_arg_N, i64 noundef %_arg_K, float noundef %_arg_Initvalue) {
entry:
  %matrixC = tail call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstruct(float %_arg_Initvalue)
  %matrixA = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 48, 0) @_Z32__spirv_CooperativeMatrixLoadKHR_1(ptr addrspace(1) noundef %_arg_accA, i32 noundef 0, i64 noundef %_arg_K, i32 noundef 1)
  %matrixB = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 2, 48, 12, 1) @_Z32__spirv_CooperativeMatrixLoadKHR_1(ptr addrspace(1) noundef %_arg_accB, i32 noundef 1, i64 noundef %_arg_K)
  %res = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z34__spirv_CooperativeMatrixMulAddKHR(target("spirv.CooperativeMatrixKHR", float, 3, 12, 48, 0) noundef %matrixA, target("spirv.CooperativeMatrixKHR", float, 2, 48, 12, 1) noundef %matrixB, target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) noundef %matrixC, i32 noundef 32)
  tail call spir_func void @_Z33__spirv_CooperativeMatrixStoreKHR(ptr addrspace(1) noundef %_arg_accC, target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) noundef %res, i32 noundef 0, i64 noundef %_arg_N, i32 noundef 1)
  ret void
}

declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstruct(float noundef)

declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z34__spirv_CooperativeMatrixMulAddKHR(target("spirv.CooperativeMatrixKHR", float, 3, 12, 48, 0) noundef, target("spirv.CooperativeMatrixKHR", float, 2, 48, 12, 1) noundef, target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) noundef, i32 noundef)

declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0) @_Z32__spirv_CooperativeMatrixLoadKHR_1(ptr addrspace(3), i32, i64, i32)

declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 48, 12, 1) @_Z32__spirv_CooperativeMatrixLoadKHR_2(ptr addrspace(3), i32, i64)

declare dso_local spir_func void @_Z33__spirv_CooperativeMatrixStoreKHR(ptr addrspace(3), target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), i32, i64, i32)
