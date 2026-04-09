; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_int4,+SPV_KHR_cooperative_matrix %s -o - | FileCheck %s
; RUNx: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_int4,+SPV_KHR_cooperative_matrix %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: Capability Int4TypeINTEL
; CHECK-DAG: Capability CooperativeMatrixKHR
; CHECK-DAG: Extension "SPV_INTEL_int4"
; CHECK-DAG: Capability Int4CooperativeMatrixINTEL
; CHECK-DAG: Extension "SPV_KHR_cooperative_matrix"

; CHECK: %[[#Int4Ty:]] = OpTypeInt 4 0
; CHECK: %[[#CoopMatTy:]] = OpTypeCooperativeMatrixKHR %[[#Int4Ty]]
; CHECK: CompositeConstruct %[[#CoopMatTy]]

define spir_kernel void @foo() {
entry:
  %call.i.i = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i4, 3, 12, 12, 2) @_Z26__spirv_CompositeConstruct(i4 noundef 0)
  ret void
}

declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", i4, 3, 12, 12, 2) @_Z26__spirv_CompositeConstruct(i4 noundef)
