; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_cooperative_matrix %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: LLVM ERROR: PackedINTEL layout require the following SPIR-V extension: SPV_INTEL_joint_matrix

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_INTEL_joint_matrix %s -o - | FileCheck %s

; CHECK-DAG: Capability CooperativeMatrixKHR
; CHECK-DAG: Capability PackedCooperativeMatrixINTEL
; CHECK-DAG: Capability CooperativeMatrixCheckedInstructionsINTEL
; CHECK-DAG: Extension "SPV_KHR_cooperative_matrix"
; CHECK-DAG: Extension "SPV_INTEL_joint_matrix"
; CHECK-DAG: %[[#Int32Ty:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Const2:]] = OpConstant %[[#Int32Ty]] 2

; CHECK: OpCooperativeMatrixLoadKHR %[[#]] %[[#]] %[[#Const2]]
; CHECK: OpCooperativeMatrixLoadKHR %[[#]] %[[#]] %[[#Const2]]
; CHECK: OpCooperativeMatrixStoreKHR %[[#]] %[[#]] %[[#Const2]]
; CHECK: OpCooperativeMatrixLoadCheckedINTEL %[[#]] %[[#]] %[[#]] %[[#]] %[[#Const2]]
; CHECK: OpCooperativeMatrixLoadCheckedINTEL %[[#]] %[[#]] %[[#]] %[[#]] %[[#Const2]]
; CHECK: OpCooperativeMatrixStoreCheckedINTEL %[[#]] %[[#]] %[[#]] %[[#]] %[[#Const2]]

define weak_odr dso_local spir_kernel void @_ZTSZZ15matrix_multiply(ptr addrspace(1) noundef align 1 %_arg_accA, ptr addrspace(1) noundef align 1 %_arg_accB, ptr addrspace(1) noundef align 1 %_arg_accC, i64 noundef %_arg_N, i64 noundef %_arg_K, i32 noundef %_arg_Initvalue) {
entry:
  %matrixC = tail call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstruct(i32 %_arg_Initvalue)
  %matrixA = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i8, 3, 12, 48, 0) @_Z32__spirv_CooperativeMatrixLoadKHR_1(ptr addrspace(1) noundef %_arg_accA, i32 noundef 2, i64 noundef %_arg_K, i32 noundef 1)
  %matrixB = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i8, 2, 48, 12, 1) @_Z32__spirv_CooperativeMatrixLoadKHR_1(ptr addrspace(1) noundef %_arg_accB, i32 noundef 2, i64 noundef %_arg_K)
  %res = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z34__spirv_CooperativeMatrixMulAddKHR(target("spirv.CooperativeMatrixKHR", i8, 3, 12, 48, 0) noundef %matrixA, target("spirv.CooperativeMatrixKHR", i8, 2, 48, 12, 1) noundef %matrixB, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef %matrixC, i32 noundef 12)
  tail call spir_func void @_Z33__spirv_CooperativeMatrixStoreKHR(ptr addrspace(1) noundef %_arg_accC, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef %res, i32 noundef 2, i64 noundef %_arg_N, i32 noundef 1)
  ret void
}

define weak_odr dso_local spir_kernel void @_ZTSZZ23matrix_multiply_checked(ptr addrspace(1) noundef align 1 %_arg_accA, ptr addrspace(1) noundef align 1 %_arg_accB, ptr addrspace(1) noundef align 1 %_arg_accC, i64 noundef %_arg_N, i64 noundef %_arg_K, i32 noundef %_arg_Initvalue) {
entry:
  %matrixC = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z46__spirv_CooperativeMatrixConstructCheckedINTEL(i32 noundef 4, i32 noundef 4, i32 noundef 12, i32 noundef 12, i32 noundef %_arg_Initvalue)
  %matrixA = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i8, 3, 12, 48, 0) @_Z41__spirv_CooperativeMatrixLoadCheckedINTEL_1(ptr addrspace(1) noundef %_arg_accA, i32 noundef 0, i32 noundef 0, i32 noundef 2, i32 noundef 12, i32 noundef 48, i64 noundef %_arg_K, i32 noundef 1)
  %matrixB = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i8, 2, 48, 12, 1) @_Z41__spirv_CooperativeMatrixLoadCheckedINTEL_2(ptr addrspace(1) noundef %_arg_accB, i32 noundef 0, i32 noundef 0, i32 noundef 2, i32 noundef 48, i32 noundef 12, i64 noundef %_arg_K)
  %res = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z34__spirv_CooperativeMatrixMulAddKHR(target("spirv.CooperativeMatrixKHR", i8, 3, 12, 48, 0) noundef %matrixA, target("spirv.CooperativeMatrixKHR", i8, 2, 48, 12, 1) noundef %matrixB, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef %matrixC, i32 noundef 12)
  tail call spir_func void @_Z42__spirv_CooperativeMatrixStoreCheckedINTEL(ptr addrspace(1) noundef %_arg_accC, i32 noundef 0, i32 noundef 0, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef %res, i32 noundef 2, i32 noundef 12, i32 noundef 12, i64 noundef %_arg_N, i32 noundef 1)
  ret void
}

declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstruct(i32)

declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z46__spirv_CooperativeMatrixConstructCheckedINTEL(i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef)

declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", i8, 3, 12, 48, 0) @_Z41__spirv_CooperativeMatrixLoadCheckedINTEL_1(ptr addrspace(4) noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i64 noundef, i32 noundef)

declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", i8, 2, 48, 12, 1) @_Z41__spirv_CooperativeMatrixLoadCheckedINTEL_2(ptr addrspace(4) noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i64 noundef)

declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z34__spirv_CooperativeMatrixMulAddKHR(target("spirv.CooperativeMatrixKHR", i8, 3, 12, 48, 0) noundef, target("spirv.CooperativeMatrixKHR", i8, 2, 48, 12, 1) noundef, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef, i32 noundef)

declare dso_local spir_func void @_Z42__spirv_CooperativeMatrixStoreCheckedINTEL(ptr addrspace(4) noundef, i32 noundef, i32 noundef, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef, i32 noundef, i32 noundef, i32 noundef, i64 noundef, i32 noundef)

declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0) @_Z32__spirv_CooperativeMatrixLoadKHR_1(ptr addrspace(3), i32, i64, i32)

declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 48, 12, 1) @_Z32__spirv_CooperativeMatrixLoadKHR_2(ptr addrspace(3), i32, i64)

declare dso_local spir_func void @_Z33__spirv_CooperativeMatrixStoreKHR(ptr addrspace(3), target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), i32, i64, i32)
