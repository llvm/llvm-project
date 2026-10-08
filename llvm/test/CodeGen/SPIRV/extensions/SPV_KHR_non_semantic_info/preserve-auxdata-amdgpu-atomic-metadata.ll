; AMDGPU atomic metadata is preserved as AuxData InstructionMetadata records.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction \
; RUN:   -spirv-preserve-auxdata \
; RUN:   %s -o - | FileCheck %s

; AMD triples preserve it without -spirv-preserve-auxdata.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s

; The old UserSemantic decoration encoding is no longer emitted.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa %s -o - \
; RUN:   | FileCheck %s --check-prefix=NOUS
; NOUS-NOT: UserSemantic

; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info %s -o - -filetype=obj \
; RUN:   | spirv-val %}

; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction \
; RUN:   -spirv-preserve-auxdata \
; RUN:   %s -o - -filetype=obj | spirv-val %}

; Records forward-reference their target, which requires
; SPV_KHR_relaxed_extended_instruction; without it they are dropped.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa \
; RUN:   --spirv-ext=-SPV_KHR_relaxed_extended_instruction \
; RUN:   -spirv-preserve-auxdata %s -o - | FileCheck %s --check-prefix=NORELAXED
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa \
; RUN:   --spirv-ext=-SPV_KHR_relaxed_extended_instruction \
; RUN:   -spirv-preserve-auxdata %s -o - -filetype=obj | spirv-val %}

; NORELAXED-NOT: SPV_KHR_relaxed_extended_instruction
; NORELAXED-NOT: amdgpu.no.fine.grained.memory
; NORELAXED-NOT: amdgpu.no.remote.memory
; NORELAXED-NOT: atomic.ignore.denormal.mode
; NORELAXED-NOT: OpExtInstWithForwardRefsKHR

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
