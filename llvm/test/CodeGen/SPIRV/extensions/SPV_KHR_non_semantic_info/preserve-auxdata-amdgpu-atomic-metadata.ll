; AMDGPU atomic metadata is preserved as AuxData InstructionMetadata records.
; AMD triples enable all extensions by default; only disabling them is tested.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa \
; RUN:   -spirv-preserve-auxdata %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa \
; RUN:   -spirv-preserve-auxdata %s -o - -filetype=obj | spirv-val %}

; AMD triples preserve it without -spirv-preserve-auxdata. The old UserSemantic
; decoration encoding is no longer emitted.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa %s -o - \
; RUN:   | FileCheck %s --implicit-check-not=UserSemantic
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - \
; RUN:   -filetype=obj | spirv-val %}

; Without SPV_KHR_non_semantic_info, nothing is emitted.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa \
; RUN:   --spirv-ext=-SPV_KHR_non_semantic_info %s -o - \
; RUN:   | FileCheck %s --check-prefixes=DROPPED,NONSI
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa \
; RUN:   --spirv-ext=-SPV_KHR_non_semantic_info %s -o - -filetype=obj \
; RUN:   | spirv-val %}

; Records forward-reference their target, which requires
; SPV_KHR_relaxed_extended_instruction; without it they are dropped.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa \
; RUN:   --spirv-ext=-SPV_KHR_relaxed_extended_instruction \
; RUN:   -spirv-preserve-auxdata %s -o - \
; RUN:   | FileCheck %s --check-prefixes=DROPPED,NORELAXED
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa \
; RUN:   --spirv-ext=-SPV_KHR_relaxed_extended_instruction \
; RUN:   -spirv-preserve-auxdata %s -o - -filetype=obj | spirv-val %}

; NONSI-NOT: NonSemantic.AuxData
; NORELAXED-NOT: SPV_KHR_relaxed_extended_instruction
; DROPPED-NOT: amdgpu.no.fine.grained.memory
; DROPPED-NOT: amdgpu.no.remote.memory
; DROPPED-NOT: atomic.ignore.denormal.mode
; DROPPED-NOT: OpExtInstWithForwardRefsKHR

; CHECK-DAG: OpExtension "SPV_KHR_relaxed_extended_instruction"
; CHECK-DAG: %[[#auxset:]] = OpExtInstImport "NonSemantic.AuxData"
; CHECK-DAG: %[[#md_nfg:]] = OpString "amdgpu.no.fine.grained.memory"
; CHECK-DAG: %[[#md_nrm:]] = OpString "amdgpu.no.remote.memory"
; CHECK-DAG: %[[#md_idn:]] = OpString "atomic.ignore.denormal.mode"
; CHECK-DAG: %[[#void:]] = OpTypeVoid

; CHECK-DAG: %[[#]] = OpExtInstWithForwardRefsKHR %[[#void]] %[[#auxset]] {{.+}} %[[#add_res:]] %[[#md_nfg]]
; CHECK-DAG: %[[#]] = OpExtInstWithForwardRefsKHR %[[#void]] %[[#auxset]] {{.+}} %[[#add_res]] %[[#md_nrm]]

; CHECK-DAG: %[[#]] = OpExtInstWithForwardRefsKHR %[[#void]] %[[#auxset]] {{.+}} %[[#fadd_res:]] %[[#md_nfg]]
; CHECK-DAG: %[[#]] = OpExtInstWithForwardRefsKHR %[[#void]] %[[#auxset]] {{.+}} %[[#fadd_res]] %[[#md_nrm]]
; CHECK-DAG: %[[#]] = OpExtInstWithForwardRefsKHR %[[#void]] %[[#auxset]] {{.+}} %[[#fadd_res]] %[[#md_idn]]

; CHECK-DAG: %[[#]] = OpExtInstWithForwardRefsKHR %[[#void]] %[[#auxset]] {{.+}} %[[#xchg_res:]] %[[#md_nfg]]

; CHECK-DAG: %[[#add_res]] = OpAtomicIAdd
; CHECK-DAG: %[[#fadd_res]] = OpAtomicFAddEXT
; CHECK-DAG: %[[#xchg_res]] = OpAtomicExchange

define amdgpu_kernel void @test_iadd(ptr addrspace(1) %ptr) {
  %val = atomicrmw add ptr addrspace(1) %ptr, i32 1 syncscope("agent") monotonic, !amdgpu.no.fine.grained.memory !0, !amdgpu.no.remote.memory !0
  ret void
}

define amdgpu_kernel void @test_fadd(ptr addrspace(1) %ptr) {
  %val = atomicrmw fadd ptr addrspace(1) %ptr, float 1.0 syncscope("agent") monotonic, !amdgpu.no.fine.grained.memory !0, !amdgpu.no.remote.memory !0, !atomic.ignore.denormal.mode !0
  ret void
}

define amdgpu_kernel void @test_xchg(ptr addrspace(1) %ptr) {
  %val = atomicrmw xchg ptr addrspace(1) %ptr, i32 1 syncscope("agent") monotonic, !amdgpu.no.fine.grained.memory !0
  ret void
}

!0 = !{}
