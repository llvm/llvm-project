; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_INTEL_joint_matrix %s -o - | FileCheck %s

; CHECK-DAG: Capability CooperativeMatrixKHR
; CHECK-DAG: Capability CooperativeMatrixInvocationInstructionsINTEL
; CHECK-DAG: Extension "SPV_KHR_cooperative_matrix"
; CHECK-DAG: Extension "SPV_INTEL_joint_matrix"
; CHECK-DAG: %[[#MatrixTy:]] = OpTypeCooperativeMatrixKHR

; CHECK: %[[#Matrix:]] = OpCompositeConstruct %[[#MatrixTy]]
; CHECK: %[[#]] = OpCooperativeMatrixGetElementCoordINTEL %[[#]] %[[#Matrix]] %[[#]]

; ModuleID = 'test-matrix-opaque.bc'
source_filename = "matrix-int8-test.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

$_ZTSZZ15matrix_multiply = comdat any

$_ZTSZZ23matrix_multiply_checked = comdat any

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTSZZ15matrix_multiply(i32 noundef %_idx, i32 noundef %_arg_Initvalue) local_unnamed_addr #0 comdat {
entry:
  %matrixC = tail call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstruct(float 0.0)
  %coord = tail call spir_func <2 x i32> @_Z45__spirv_CooperativeMatrixGetElementCoordINTEL(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %matrixC, i32 noundef %_idx)
  ret void
}
; Function Attrs: convergent
declare dso_local spir_func <2 x i32> @_Z45__spirv_CooperativeMatrixGetElementCoordINTEL(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), i32 noundef)

; Function Attrs: convergent
declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z26__spirv_CompositeConstruct(i32 noundef) local_unnamed_addr #1

attributes #0 = { convergent norecurse "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="matrix-int8-test.cpp" "uniform-work-group-size"="true" }
attributes #1 = { convergent }
