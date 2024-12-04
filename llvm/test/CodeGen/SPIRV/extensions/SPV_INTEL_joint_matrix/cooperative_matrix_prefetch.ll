; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_INTEL_joint_matrix %s -o - | FileCheck %s

; CHECK-DAG: Capability CooperativeMatrixPrefetchINTEL
; CHECK-DAG: Extension "SPV_KHR_cooperative_matrix"
; CHECK-DAG: OpExtension "SPV_INTEL_joint_matrix"

; CHECK-DAG: OpCooperativeMatrixPrefetchINTEL %[[#]] %[[#]] %[[#]] 0 %[[#]]
; CHECK-DAG: OpCooperativeMatrixPrefetchINTEL %[[#]] %[[#]] %[[#]] 0 %[[#]] %[[#]]
; CHECK-DAG: OpCooperativeMatrixPrefetchINTEL %[[#]] %[[#]] %[[#]] 0 %[[#]] %[[#]] 1

; ModuleID = 'test-matrix-opaque.bc'
source_filename = "matrix-int8-test.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

$_ZTSZZ15matrix_multiply = comdat any

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTSZZ15matrix_multiply(ptr addrspace(1) noundef align 1 %_arg_accA, ptr addrspace(1) noundef align 1 %_arg_accB, ptr addrspace(1) noundef align 1 %_arg_accC, i64 noundef %_arg_K) local_unnamed_addr #0 comdat {
entry:
  tail call spir_func void @_Z38__spirv_CooperativeMatrixPrefetchINTELPU3AS4ciiii(ptr addrspace(1) noundef %_arg_accA, i32 noundef 12, i32 noundef 48, i32 noundef 0, i32 noundef 0) #1
  tail call spir_func void @_Z38__spirv_CooperativeMatrixPrefetchINTELPU3AS4ciiiil(ptr addrspace(1) noundef %_arg_accB, i32 noundef 12, i32 noundef 48, i32 noundef 0, i32 noundef 0, i64 noundef %_arg_K) #1
  tail call spir_func void @_Z38__spirv_CooperativeMatrixPrefetchINTELPU3AS4ciiiili(ptr addrspace(1) noundef %_arg_accC, i32 noundef 12, i32 noundef 48, i32 noundef 0, i32 noundef 0, i64 noundef %_arg_K, i32 1) #1
  ret void
}

declare dso_local spir_func void @_Z38__spirv_CooperativeMatrixPrefetchINTELPU3AS4ciiii(ptr addrspace(1) noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #1

declare dso_local spir_func void @_Z38__spirv_CooperativeMatrixPrefetchINTELPU3AS4ciiiil(ptr addrspace(1) noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i64 noundef) local_unnamed_addr #1

declare dso_local spir_func void @_Z38__spirv_CooperativeMatrixPrefetchINTELPU3AS4ciiiili(ptr addrspace(1) noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i64 noundef, i32 noundef) local_unnamed_addr #1

attributes #0 = { convergent norecurse "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="matrix-int8-test.cpp" "uniform-work-group-size"="true" }
attributes #1 = { convergent }
