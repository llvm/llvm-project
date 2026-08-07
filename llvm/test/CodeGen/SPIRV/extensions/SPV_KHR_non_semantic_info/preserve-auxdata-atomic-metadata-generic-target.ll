; NonSemantic.AuxData is non-semantic by construction: a consumer that does not
; understand the extended instruction set ignores it. Preserving instruction
; metadata is therefore not restricted to AMD targets, even though the metadata
; names that currently benefit are amdgpu.*. Verify that -spirv-preserve-auxdata
; emits it on a generic target too.
;
; The AMD-target behaviour and the full set of metadata kinds are covered by
; preserve-auxdata-amdgpu-atomic-metadata.ll.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info -spirv-preserve-auxdata \
; RUN:   %s -o - | FileCheck %s

; Without the option nothing is emitted, on any target.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info %s -o - \
; RUN:   | FileCheck %s --check-prefix=OFF

; OFF-NOT: amdgpu.no.fine.grained.memory

; CHECK-DAG: %[[#auxset:]] = OpExtInstImport "NonSemantic.AuxData"
; CHECK-DAG: %[[#md_nfg:]] = OpString "amdgpu.no.fine.grained.memory"
; CHECK-DAG: %[[#void:]] = OpTypeVoid
; CHECK-DAG: %[[#]] = OpExtInst %[[#void]] %[[#auxset]] {{.+}} %[[#add_res:]] %[[#md_nfg]]
; CHECK-DAG: %[[#add_res]] = OpAtomicIAdd

; No spirv-val run with the option on: the AuxData instruction forward-references
; the atomic's result <id>, which spirv-val rejects. See the CHECK-INVALID pin in
; preserve-auxdata-amdgpu-atomic-metadata.ll.

define spir_func void @test_iadd(ptr addrspace(1) %ptr) {
  %val = atomicrmw add ptr addrspace(1) %ptr, i32 1 monotonic, !amdgpu.no.fine.grained.memory !0
  ret void
}

!0 = !{}
