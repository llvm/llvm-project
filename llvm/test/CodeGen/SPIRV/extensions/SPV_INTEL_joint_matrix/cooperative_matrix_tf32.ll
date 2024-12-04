; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_INTEL_joint_matrix %s -o - | FileCheck %s

; CHECK-DAG: Capability CooperativeMatrixKHR
; CHECK-DAG: Extension "SPV_KHR_cooperative_matrix"
; CHECK-DAG: Extension "SPV_INTEL_joint_matrix"
; CHECK-DAG: OpCapability CooperativeMatrixTF32ComponentTypeINTEL

; CHECK: OpCooperativeMatrixMulAddKHR %[[#]] %[[#]] %[[#]] %[[#]] MatrixAAndBTF32ComponentsINTEL

; ModuleID = 'test-matrix-opaque.bc'
source_filename = "matrix-int8-test.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

$_ZTSZZ15matrix_multiply = comdat any

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTSZZ15matrix_multiply(ptr addrspace(1) noundef align 1 %_arg_accA, ptr addrspace(1) noundef align 1 %_arg_accB, ptr addrspace(1) noundef align 1 %_arg_accC, i64 noundef %_arg_N, i64 noundef %_arg_K, float noundef %_arg_Initvalue) local_unnamed_addr #0 comdat {
entry:
  %matrixC = tail call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstruct(float %_arg_Initvalue)
  %matrixA = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 48, 0) @_Z32__spirv_CooperativeMatrixLoadKHR_1(ptr addrspace(1) noundef %_arg_accA, i32 noundef 2, i64 noundef %_arg_K, i32 noundef 1) #2
  %matrixB = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 2, 48, 12, 1) @_Z32__spirv_CooperativeMatrixLoadKHR_1(ptr addrspace(1) noundef %_arg_accB, i32 noundef 2, i64 noundef %_arg_K) #2
  %res = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z34__spirv_CooperativeMatrixMulAddKHR(target("spirv.CooperativeMatrixKHR", float, 3, 12, 48, 0) noundef %matrixA, target("spirv.CooperativeMatrixKHR", float, 2, 48, 12, 1) noundef %matrixB, target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) noundef %matrixC, i32 noundef 32) #2
  tail call spir_func void @_Z33__spirv_CooperativeMatrixStoreKHR(ptr addrspace(1) noundef %_arg_accC, target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) noundef %res, i32 noundef 2, i64 noundef %_arg_N, i32 noundef 1) #2
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstruct(float noundef) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z34__spirv_CooperativeMatrixMulAddKHR(target("spirv.CooperativeMatrixKHR", float, 3, 12, 48, 0) noundef, target("spirv.CooperativeMatrixKHR", float, 2, 48, 12, 1) noundef, target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) noundef, i32 noundef) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0) @_Z32__spirv_CooperativeMatrixLoadKHR_1(ptr addrspace(3), i32, i64, i32) #1

; Function Attrs: convergent
declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 48, 12, 1) @_Z32__spirv_CooperativeMatrixLoadKHR_2(ptr addrspace(3), i32, i64) #1

; Function Attrs: convergent
declare dso_local spir_func void @_Z33__spirv_CooperativeMatrixStoreKHR(ptr addrspace(3), target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), i32, i64, i32) #1


attributes #0 = { convergent norecurse "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="matrix-int8-test.cpp" "uniform-work-group-size"="true" }
attributes #1 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent }
