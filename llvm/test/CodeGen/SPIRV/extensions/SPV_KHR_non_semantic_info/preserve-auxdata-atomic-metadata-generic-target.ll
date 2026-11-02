; Instruction metadata AuxData also works on non-AMD triples.
; Full coverage is in preserve-auxdata-amdgpu-atomic-metadata.ll.

; Non-AMD triple: the flag and extensions must be passed explicitly.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction \
; RUN:   -spirv-preserve-auxdata %s -o - | FileCheck %s

; Without -spirv-preserve-auxdata, nothing is emitted.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info %s -o - \
; RUN:   | FileCheck %s --check-prefix=OFF

; OFF-NOT: amdgpu.no.fine.grained.memory

; Without SPV_KHR_relaxed_extended_instruction, records are dropped.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info -spirv-preserve-auxdata %s -o - \
; RUN:   | FileCheck %s --check-prefix=NORELAXED
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 \
; RUN:   -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info \
; RUN:   -spirv-preserve-auxdata %s -o - -filetype=obj | spirv-val %}

; NORELAXED-NOT: SPV_KHR_relaxed_extended_instruction
; NORELAXED-NOT: amdgpu.no.fine.grained.memory
; NORELAXED-NOT: OpExtInstWithForwardRefsKHR

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
