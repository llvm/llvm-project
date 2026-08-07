; Test that atomicrmw uinc_wrap/udec_wrap with AMDGPU metadata emit both
; OpFunctionCall (for the atomic) and AuxData InstructionMetadata (for the
; metadata), all gated on -spirv-preserve-auxdata.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info -spirv-preserve-auxdata \
; RUN:   %s -o - | FileCheck %s

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info %s -o - \
; RUN:   | FileCheck %s --check-prefix=OFF

; OFF-NOT: amdgpu.no.fine.grained.memory
; OFF-NOT: amdgpu.no.remote.memory

; Default output, with the feature off, validates.
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 \
; RUN:   -mtriple=spirv64-amd-amdhsa --spirv-ext=+SPV_KHR_non_semantic_info \
; RUN:   %s -o - -filetype=obj | spirv-val %}

; As in preserve-auxdata-amdgpu-atomic-metadata.ll, the AuxData instructions
; sit in the module-level section and forward-reference a result <id> defined
; inside a function body -- here the OpFunctionCall standing in for the
; atomicrmw. spirv-val does not accept that, so pin the rejection rather than
; leave the module unvalidated. Drop the "not" and the CHECK-INVALID prefix
; when the forward reference is resolved.
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 \
; RUN:   -mtriple=spirv64-amd-amdhsa --spirv-ext=+SPV_KHR_non_semantic_info \
; RUN:   -spirv-preserve-auxdata %s -o - -filetype=obj | not spirv-val 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-INVALID %}

; CHECK-INVALID: has not been defined

; CHECK-DAG: %[[#auxset:]] = OpExtInstImport "NonSemantic.AuxData"
; CHECK-DAG: %[[#md_nfg:]] = OpString "amdgpu.no.fine.grained.memory"
; CHECK-DAG: %[[#md_nrm:]] = OpString "amdgpu.no.remote.memory"
; CHECK-DAG: %[[#void:]] = OpTypeVoid

; CHECK-DAG: OpDecorate %[[#UIncFn:]] LinkageAttributes "__translate_spirv_atomic_uinc_wrap_p1_i32" Import
; CHECK-DAG: OpDecorate %[[#UDecFn:]] LinkageAttributes "__translate_spirv_atomic_udec_wrap_p1_i32" Import

; AuxData for the uinc_wrap result.
; CHECK-DAG: %[[#]] = OpExtInst %[[#void]] %[[#auxset]] {{.+}} %[[#uinc_res:]] %[[#md_nfg]]
; CHECK-DAG: %[[#]] = OpExtInst %[[#void]] %[[#auxset]] {{.+}} %[[#uinc_res]] %[[#md_nrm]]

; AuxData for the udec_wrap result.
; CHECK-DAG: %[[#]] = OpExtInst %[[#void]] %[[#auxset]] {{.+}} %[[#udec_res:]] %[[#md_nfg]]

; The function calls themselves.
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
