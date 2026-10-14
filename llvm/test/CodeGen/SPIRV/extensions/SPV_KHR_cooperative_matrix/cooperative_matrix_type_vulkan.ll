; Check that the opaque CooperativeMatrix type under the Vulkan/Shader flavor,
; target("spirv.CooperativeMatrixKHR", elem, scope, rows, cols, use), lowers to
; OpTypeCooperativeMatrixKHR, gated behind the CooperativeMatrixKHR capability and
; the SPV_KHR_cooperative_matrix extension, and that it errors cleanly without it.
; This is a text-emission check only; the type is forced via a by-value parameter,
; so it is not run through spirv-val here.

; RUN: not llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_KHR_vulkan_memory_model %s -o - | FileCheck %s

; CHECK-ERROR: LLVM ERROR: OpTypeCooperativeMatrixKHR type requires the following SPIR-V extension: SPV_KHR_cooperative_matrix

; CHECK-DAG: OpCapability CooperativeMatrixKHR
; CHECK-DAG: OpExtension "SPV_KHR_cooperative_matrix"
; CHECK-DAG: {{%[0-9]+}} = OpTypeCooperativeMatrixKHR

; A by-value parameter of the opaque type forces OpTypeCooperativeMatrixKHR into the
; module (referenced by OpTypeFunction, so it is not eliminated like an unused local).
define spir_func void @use_coopmat(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %m) {
entry:
  ret void
}
