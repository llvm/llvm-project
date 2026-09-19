; NonSemantic.AuxData is non-semantic by construction: a consumer that does not
; understand the extended instruction set ignores it. Preserving instruction
; metadata is therefore not restricted to AMD targets, even though the metadata
; names that currently benefit are amdgpu.*. Verify that -spirv-preserve-auxdata
; emits it on a generic target too.
;
; The AMD-target behaviour and the full set of metadata kinds are covered by
; preserve-auxdata-amdgpu-atomic-metadata.ll.

; The forward-referencing metadata needs SPV_KHR_relaxed_extended_instruction,
; which a generic triple (unlike AMD) does not auto-enable, so request it.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction \
; RUN:   -spirv-preserve-auxdata %s -o - | FileCheck %s

; Without the option nothing is emitted, on any target.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info %s -o - \
; RUN:   | FileCheck %s --check-prefix=OFF

; OFF-NOT: amdgpu.no.fine.grained.memory

; CHECK-DAG: OpExtension "SPV_KHR_relaxed_extended_instruction"
; CHECK-DAG: %[[#auxset:]] = OpExtInstImport "NonSemantic.AuxData"
; CHECK-DAG: %[[#md_nfg:]] = OpString "amdgpu.no.fine.grained.memory"
; CHECK-DAG: %[[#void:]] = OpTypeVoid
; CHECK-DAG: %[[#]] = OpExtInstWithForwardRefsKHR %[[#void]] %[[#auxset]] {{.+}} %[[#add_res:]] %[[#md_nfg]]
; CHECK-DAG: %[[#add_res]] = OpAtomicIAdd

; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 \
; RUN:   -mtriple=spirv64-unknown-unknown \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction \
; RUN:   -spirv-preserve-auxdata %s -o - -filetype=obj | spirv-val %}

define spir_func void @test_iadd(ptr addrspace(1) %ptr) {
  %val = atomicrmw add ptr addrspace(1) %ptr, i32 1 monotonic, !amdgpu.no.fine.grained.memory !0
  ret void
}

!0 = !{}
