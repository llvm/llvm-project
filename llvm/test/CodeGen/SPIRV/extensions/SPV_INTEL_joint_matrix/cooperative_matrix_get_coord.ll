; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_cooperative_matrix %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: LLVM ERROR: OpCooperativeMatrixGetElementCoordINTEL requires the following SPIR-V extension: SPV_INTEL_joint_matrix

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_INTEL_joint_matrix %s -o - | FileCheck %s

; CHECK-DAG: Capability CooperativeMatrixKHR
; CHECK-DAG: Capability CooperativeMatrixInvocationInstructionsINTEL
; CHECK-DAG: Extension "SPV_KHR_cooperative_matrix"
; CHECK-DAG: Extension "SPV_INTEL_joint_matrix"
; CHECK-DAG: %[[#MatrixTy:]] = OpTypeCooperativeMatrixKHR

; CHECK: %[[#Matrix:]] = OpCompositeConstruct %[[#MatrixTy]]
; CHECK: %[[#]] = OpCooperativeMatrixGetElementCoordINTEL %[[#]] %[[#Matrix]] %[[#]]

define weak_odr dso_local spir_kernel void @_ZTSZZ15matrix_multiply(i32 noundef %_idx, i32 noundef %_arg_Initvalue) {
entry:
  %matrixC = tail call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstruct(float 0.0)
  %coord = tail call spir_func <2 x i32> @_Z45__spirv_CooperativeMatrixGetElementCoordINTEL(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %matrixC, i32 noundef %_idx)
  ret void
}
declare dso_local spir_func <2 x i32> @_Z45__spirv_CooperativeMatrixGetElementCoordINTEL(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), i32 noundef)

declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstruct(i32 noundef) local_unnamed_addr
