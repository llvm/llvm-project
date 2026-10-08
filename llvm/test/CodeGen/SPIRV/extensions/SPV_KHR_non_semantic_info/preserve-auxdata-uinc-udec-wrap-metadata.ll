; uinc_wrap/udec_wrap lower to helper OpFunctionCalls; their AMDGPU metadata
; is preserved as AuxData InstructionMetadata on the call result.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction \
; RUN:   -spirv-preserve-auxdata %s -o - | FileCheck %s

; AMD triples preserve it without -spirv-preserve-auxdata.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s

; The old UserSemantic decoration encoding is no longer emitted.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa %s -o - \
; RUN:   | FileCheck %s --check-prefix=NOUS
; NOUS-NOT: UserSemantic

; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 \
; RUN:   -mtriple=spirv64-amd-amdhsa --spirv-ext=+SPV_KHR_non_semantic_info \
; RUN:   %s -o - -filetype=obj | spirv-val %}

; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 \
; RUN:   -mtriple=spirv64-amd-amdhsa \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction \
; RUN:   -spirv-preserve-auxdata %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpExtension "SPV_KHR_relaxed_extended_instruction"
; CHECK-DAG: %[[#auxset:]] = OpExtInstImport "NonSemantic.AuxData"
; CHECK-DAG: %[[#md_nfg:]] = OpString "amdgpu.no.fine.grained.memory"
; CHECK-DAG: %[[#md_nrm:]] = OpString "amdgpu.no.remote.memory"
; CHECK-DAG: %[[#void:]] = OpTypeVoid

; CHECK-DAG: OpDecorate %[[#UIncFn:]] LinkageAttributes "__translate_spirv_atomic_uinc_wrap_p1_i32" Import
; CHECK-DAG: OpDecorate %[[#UDecFn:]] LinkageAttributes "__translate_spirv_atomic_udec_wrap_p1_i32" Import

; CHECK-DAG: %[[#]] = OpExtInstWithForwardRefsKHR %[[#void]] %[[#auxset]] {{.+}} %[[#uinc_res:]] %[[#md_nfg]]
; CHECK-DAG: %[[#]] = OpExtInstWithForwardRefsKHR %[[#void]] %[[#auxset]] {{.+}} %[[#uinc_res]] %[[#md_nrm]]

; CHECK-DAG: %[[#]] = OpExtInstWithForwardRefsKHR %[[#void]] %[[#auxset]] {{.+}} %[[#udec_res:]] %[[#md_nfg]]

; CHECK-DAG: %[[#uinc_res]] = OpFunctionCall %[[#]] %[[#UIncFn]]
; CHECK-DAG: %[[#udec_res]] = OpFunctionCall %[[#]] %[[#UDecFn]]

@ui = common dso_local addrspace(1) global i32 0, align 4

define amdgpu_kernel void @test_uinc_wrap() {
entry:
  %uinc = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 seq_cst, !amdgpu.no.fine.grained.memory !0, !amdgpu.no.remote.memory !0
  ret void
}

define amdgpu_kernel void @test_udec_wrap() {
entry:
  %udec = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 seq_cst, !amdgpu.no.fine.grained.memory !0
  ret void
}

!0 = !{}
