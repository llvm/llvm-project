; Test that AMDGPU atomic metadata is preserved as NonSemantic.AuxData
; InstructionMetadata (opcode 5).

; Positive: with -spirv-preserve-auxdata, metadata emitted as AuxData.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction \
; RUN:   -spirv-preserve-auxdata \
; RUN:   %s -o - | FileCheck %s

; Negative: without -spirv-preserve-auxdata, no metadata strings.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info %s -o - \
; RUN:   | FileCheck %s --check-prefix=OFF

; OFF-NOT: amdgpu.no.fine.grained.memory
; OFF-NOT: amdgpu.no.remote.memory
; OFF-NOT: amdgpu.ignore.denormal.mode

; Default output, with the feature off, validates.
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info %s -o - -filetype=obj \
; RUN:   | spirv-val %}

; InstructionMetadata forward-references the atomic's result <id>, encoded as
; OpExtInstWithForwardRefsKHR. Upstream spirv-val only permits forward refs from
; debug-info sets, so it still rejects this. Drop the "not"/CHECK-INVALID once
; https://github.com/KhronosGroup/SPIRV-Tools/pull/6847 lands.
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction \
; RUN:   -spirv-preserve-auxdata \
; RUN:   %s -o - -filetype=obj | not spirv-val 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-INVALID %}

; CHECK-INVALID: has not been defined

; CHECK-DAG: OpExtension "SPV_KHR_relaxed_extended_instruction"
; CHECK-DAG: %[[#auxset:]] = OpExtInstImport "NonSemantic.AuxData"
; CHECK-DAG: %[[#md_nfg:]] = OpString "amdgpu.no.fine.grained.memory"
; CHECK-DAG: %[[#md_nrm:]] = OpString "amdgpu.no.remote.memory"
; CHECK-DAG: %[[#md_idn:]] = OpString "amdgpu.ignore.denormal.mode"
; CHECK-DAG: %[[#void:]] = OpTypeVoid

; Integer atomic (add) with two metadata kinds.
; CHECK-DAG: %[[#]] = OpExtInstWithForwardRefsKHR %[[#void]] %[[#auxset]] {{.+}} %[[#add_res:]] %[[#md_nfg]]
; CHECK-DAG: %[[#]] = OpExtInstWithForwardRefsKHR %[[#void]] %[[#auxset]] {{.+}} %[[#add_res]] %[[#md_nrm]]

; Float atomic (fadd) with all three metadata kinds.
; CHECK-DAG: %[[#]] = OpExtInstWithForwardRefsKHR %[[#void]] %[[#auxset]] {{.+}} %[[#fadd_res:]] %[[#md_nfg]]
; CHECK-DAG: %[[#]] = OpExtInstWithForwardRefsKHR %[[#void]] %[[#auxset]] {{.+}} %[[#fadd_res]] %[[#md_nrm]]
; CHECK-DAG: %[[#]] = OpExtInstWithForwardRefsKHR %[[#void]] %[[#auxset]] {{.+}} %[[#fadd_res]] %[[#md_idn]]

; Atomic (xchg) with only one metadata kind.
; CHECK-DAG: %[[#]] = OpExtInstWithForwardRefsKHR %[[#void]] %[[#auxset]] {{.+}} %[[#xchg_res:]] %[[#md_nfg]]

; The atomic instructions themselves (forward-referenced by AuxData above).
; CHECK-DAG: %[[#add_res]] = OpAtomicIAdd
; CHECK-DAG: %[[#fadd_res]] = OpAtomicFAddEXT
; CHECK-DAG: %[[#xchg_res]] = OpAtomicExchange


define amdgpu_kernel void @test_iadd(ptr addrspace(1) %ptr) {
  %val = atomicrmw add ptr addrspace(1) %ptr, i32 1 syncscope("agent") monotonic, !amdgpu.no.fine.grained.memory !0, !amdgpu.no.remote.memory !0
  ret void
}

define amdgpu_kernel void @test_fadd(ptr addrspace(1) %ptr) {
  %val = atomicrmw fadd ptr addrspace(1) %ptr, float 1.0 syncscope("agent") monotonic, !amdgpu.no.fine.grained.memory !0, !amdgpu.no.remote.memory !0, !amdgpu.ignore.denormal.mode !0
  ret void
}

define amdgpu_kernel void @test_xchg(ptr addrspace(1) %ptr) {
  %val = atomicrmw xchg ptr addrspace(1) %ptr, i32 1 syncscope("agent") monotonic, !amdgpu.no.fine.grained.memory !0
  ret void
}

!0 = !{}
