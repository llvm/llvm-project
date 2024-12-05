; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_cooperative_matrix %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: LLVM ERROR: OpCooperativeMatrixConstructCheckedINTEL instructions require the following SPIR-V extension: SPV_INTEL_joint_matrix

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_INTEL_joint_matrix %s -o - | FileCheck %s

; CHECK-DAG: Capability CooperativeMatrixKHR
; CHECK-DAG: Capability CooperativeMatrixCheckedInstructionsINTEL
; CHECK-DAG: Extension "SPV_KHR_cooperative_matrix"
; CHECK-DAG: Extension "SPV_INTEL_joint_matrix"
; CHECK-DAG: %[[#Int8Ty:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#Int32Ty:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Const12:]] = OpConstant %[[#Int32Ty]] 12
; CHECK-DAG: %[[#Const48:]] = OpConstant %[[#Int32Ty]] 48
; CHECK-DAG: %[[#Const0:]] = OpConstant %[[#Int32Ty]] 0
; CHECK-DAG: %[[#Const3:]] = OpConstant %[[#Int32Ty]] 3
; CHECK-DAG: %[[#Const2:]] = OpConstant %[[#Int32Ty]] 2
; CHECK-DAG: %[[#Const1:]] = OpConstant %[[#Int32Ty]] 1
; CHECK-DAG: %[[#MatTy1:]] = OpTypeCooperativeMatrixKHR %[[#Int32Ty]] %[[#Const3]] %[[#Const12]] %[[#Const12]] %[[#Const2]]
; CHECK-DAG: %[[#MatTy2:]] = OpTypeCooperativeMatrixKHR %[[#Int8Ty]] %[[#Const3]] %[[#Const12]] %[[#Const48]] %[[#Const0]]
; CHECK-DAG: %[[#MatTy3:]] = OpTypeCooperativeMatrixKHR %[[#Int8Ty]] %[[#Const2]] %[[#Const48]] %[[#Const12]] %[[#Const1]]
; CHECK: OpCooperativeMatrixConstructCheckedINTEL %[[#MatTy1]]
; CHECK: OpCooperativeMatrixLoadCheckedINTEL %[[#MatTy2]]
; CHECK: OpCooperativeMatrixLoadCheckedINTEL %[[#MatTy3]]
; CHECK: OpCooperativeMatrixMulAddKHR %[[#MatTy1]]
; CHECK: OpCooperativeMatrixStoreCheckedINTEL

define weak_odr dso_local spir_kernel void @_ZTSZZ15matrix_multiply(ptr addrspace(1) noundef align 1 %_arg_accA, ptr addrspace(1) noundef align 1 %_arg_accB, ptr addrspace(1) noundef align 1 %_arg_accC, i64 noundef %_arg_N, i64 noundef %_arg_K, i32 noundef %_arg_Initvalue) {
entry:
  %matrixC = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z46__spirv_CooperativeMatrixConstructCheckedINTEL(i32 noundef 4, i32 noundef 4, i32 noundef 12, i32 noundef 12, i32 noundef %_arg_Initvalue)
  %matrixA = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i8, 3, 12, 48, 0) @_Z41__spirv_CooperativeMatrixLoadCheckedINTEL_1(ptr addrspace(1) noundef %_arg_accA, i32 noundef 0, i32 noundef 0, i32 noundef 0, i32 noundef 12, i32 noundef 48, i64 noundef %_arg_K, i32 noundef 1)
  %matrixB = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i8, 2, 48, 12, 1) @_Z41__spirv_CooperativeMatrixLoadCheckedINTEL_2(ptr addrspace(1) noundef %_arg_accB, i32 noundef 0, i32 noundef 0, i32 noundef 1, i32 noundef 48, i32 noundef 12, i64 noundef %_arg_K)
  %res = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z34__spirv_CooperativeMatrixMulAddKHR(target("spirv.CooperativeMatrixKHR", i8, 3, 12, 48, 0) noundef %matrixA, target("spirv.CooperativeMatrixKHR", i8, 2, 48, 12, 1) noundef %matrixB, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef %matrixC, i32 noundef 12)
  tail call spir_func void @_Z42__spirv_CooperativeMatrixStoreCheckedINTEL(ptr addrspace(1) noundef %_arg_accC, i32 noundef 0, i32 noundef 0, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef %res, i32 noundef 0, i32 noundef 12, i32 noundef 12, i64 noundef %_arg_N, i32 noundef 1)
  ret void
}

declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z46__spirv_CooperativeMatrixConstructCheckedINTEL(i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef)

declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", i8, 3, 12, 48, 0) @_Z41__spirv_CooperativeMatrixLoadCheckedINTEL_1(ptr addrspace(4) noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i64 noundef, i32 noundef)

declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", i8, 2, 48, 12, 1) @_Z41__spirv_CooperativeMatrixLoadCheckedINTEL_2(ptr addrspace(4) noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i64 noundef)

declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z34__spirv_CooperativeMatrixMulAddKHR(target("spirv.CooperativeMatrixKHR", i8, 3, 12, 48, 0) noundef, target("spirv.CooperativeMatrixKHR", i8, 2, 48, 12, 1) noundef, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef, i32 noundef)

declare dso_local spir_func void @_Z42__spirv_CooperativeMatrixStoreCheckedINTEL(ptr addrspace(4) noundef, i32 noundef, i32 noundef, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef, i32 noundef, i32 noundef, i32 noundef, i64 noundef, i32 noundef)
