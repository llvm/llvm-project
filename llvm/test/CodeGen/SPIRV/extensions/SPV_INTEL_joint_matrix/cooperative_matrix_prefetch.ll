; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_cooperative_matrix %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: LLVM ERROR: OpCooperativeMatrixPrefetchINTEL instruction requires the following SPIR-V extension: SPV_INTEL_joint_matrix

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_INTEL_joint_matrix %s -o - | FileCheck %s

; CHECK-DAG: Capability CooperativeMatrixPrefetchINTEL
; CHECK-DAG: Extension "SPV_KHR_cooperative_matrix"
; CHECK-DAG: OpExtension "SPV_INTEL_joint_matrix"

; CHECK-DAG: OpCooperativeMatrixPrefetchINTEL %[[#]] %[[#]] %[[#]] 0 %[[#]]
; CHECK-DAG: OpCooperativeMatrixPrefetchINTEL %[[#]] %[[#]] %[[#]] 0 %[[#]] %[[#]]
; CHECK-DAG: OpCooperativeMatrixPrefetchINTEL %[[#]] %[[#]] %[[#]] 0 %[[#]] %[[#]] 1

define weak_odr dso_local spir_kernel void @_ZTSZZ15matrix_multiply(ptr addrspace(1) noundef align 1 %_arg_accA, ptr addrspace(1) noundef align 1 %_arg_accB, ptr addrspace(1) noundef align 1 %_arg_accC, i64 noundef %_arg_K) {
entry:
  tail call spir_func void @_Z38__spirv_CooperativeMatrixPrefetchINTELPU3AS4ciiii(ptr addrspace(1) noundef %_arg_accA, i32 noundef 12, i32 noundef 48, i32 noundef 0, i32 noundef 0)
  tail call spir_func void @_Z38__spirv_CooperativeMatrixPrefetchINTELPU3AS4ciiiil(ptr addrspace(1) noundef %_arg_accB, i32 noundef 12, i32 noundef 48, i32 noundef 0, i32 noundef 0, i64 noundef %_arg_K)
  tail call spir_func void @_Z38__spirv_CooperativeMatrixPrefetchINTELPU3AS4ciiiili(ptr addrspace(1) noundef %_arg_accC, i32 noundef 12, i32 noundef 48, i32 noundef 0, i32 noundef 0, i64 noundef %_arg_K, i32 1)
  ret void
}

declare dso_local spir_func void @_Z38__spirv_CooperativeMatrixPrefetchINTELPU3AS4ciiii(ptr addrspace(1) noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef)

declare dso_local spir_func void @_Z38__spirv_CooperativeMatrixPrefetchINTELPU3AS4ciiiil(ptr addrspace(1) noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i64 noundef)

declare dso_local spir_func void @_Z38__spirv_CooperativeMatrixPrefetchINTELPU3AS4ciiiili(ptr addrspace(1) noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i64 noundef, i32 noundef)
