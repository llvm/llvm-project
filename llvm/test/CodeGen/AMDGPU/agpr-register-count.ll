; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s

; GCN-LABEL: {{^}}kernel_32_agprs:
; GCN:    .amdhsa_next_free_vgpr 32
; GCN:    NumVgprs: 9
; GCN:    NumAgprs: 32
; GCN:    TotalNumVgprs: 32
; GCN:    VGPRBlocks: 7
; GCN:    NumVGPRsForWavesPerEU: 32
; GCN:    Occupancy: 8
define amdgpu_kernel void @kernel_32_agprs() {
bb:
  call void asm sideeffect "", "~{v8}" ()
  call void asm sideeffect "", "~{a31}" ()
  ret void
}

; GCN-LABEL: {{^}}kernel_0_agprs:
; GCN:    .amdhsa_next_free_vgpr 1
; GCN:    NumVgprs: 1
; GCN:    NumAgprs: 0
; GCN:    TotalNumVgprs: 1
; GCN:    VGPRBlocks: 0
; GCN:    NumVGPRsForWavesPerEU: 1
; GCN:    Occupancy: 10
define amdgpu_kernel void @kernel_0_agprs() {
bb:
  call void asm sideeffect "", "~{v0}" ()
  ret void
}

; GCN-LABEL: {{^}}kernel_40_vgprs:
; GCN:    .amdhsa_next_free_vgpr 40
; GCN:    NumVgprs: 40
; GCN:    NumAgprs: 16
; GCN:    TotalNumVgprs: 40
; GCN:    VGPRBlocks: 9
; GCN:    NumVGPRsForWavesPerEU: 40
; GCN:    Occupancy: 6
define amdgpu_kernel void @kernel_40_vgprs() {
bb:
  call void asm sideeffect "", "~{v39}" ()
  call void asm sideeffect "", "~{a15}" ()
  ret void
}

; GCN-LABEL: {{^}}func_32_agprs:
; GCN:    NumVgprs: 9
; GCN:    NumAgprs: 32
; GCN:    TotalNumVgprs: 32
define void @func_32_agprs() #0 {
bb:
  call void asm sideeffect "", "~{v8}" ()
  call void asm sideeffect "", "~{a31}" ()
  ret void
}

; GCN-LABEL: {{^}}func_32_vgprs:
; GCN:    NumVgprs: 32
; GCN:    NumAgprs: 9
; GCN:    TotalNumVgprs: 32
define void @func_32_vgprs() {
bb:
  call void asm sideeffect "", "~{v31}" ()
  call void asm sideeffect "", "~{a8}" ()
  ret void
}

; GCN-LABEL: {{^}}func_0_agprs:
; GCN:    NumVgprs: 1
; GCN:    NumAgprs: 0
; GCN:    TotalNumVgprs: 1
define amdgpu_kernel void @func_0_agprs() {
bb:
  call void asm sideeffect "", "~{v0}" ()
  ret void
}

; GCN-LABEL: {{^}}kernel_max_gprs:
; GCN:    .amdhsa_next_free_vgpr 256
; GCN:    NumVgprs: 256
; GCN:    NumAgprs: 256
; GCN:    TotalNumVgprs: 256
; GCN:    VGPRBlocks: 63
; GCN:    NumVGPRsForWavesPerEU: 256
; GCN:    Occupancy: 1
define amdgpu_kernel void @kernel_max_gprs() {
bb:
  call void asm sideeffect "", "~{v255}" ()
  call void asm sideeffect "", "~{a255}" ()
  ret void
}

; GCN-LABEL: {{^}}kernel_call_func_32_agprs:
; GCN:    .amdhsa_next_free_vgpr 32
; GCN:    NumVgprs: 9
; GCN:    NumAgprs: 32
; GCN:    TotalNumVgprs: 32
; GCN:    VGPRBlocks: 7
; GCN:    NumVGPRsForWavesPerEU: 32
; GCN:    Occupancy: 8
define amdgpu_kernel void @kernel_call_func_32_agprs() {
bb:
  call void @func_32_agprs() #0
  ret void
}

; GCN-LABEL: {{^}}func_call_func_32_agprs:
; GCN:    NumVgprs: 9
; GCN:    NumAgprs: 32
; GCN:    TotalNumVgprs: 32
define void @func_call_func_32_agprs() {
bb:
  call void @func_32_agprs() #0
  ret void
}

declare void @undef_func()

; GCN-LABEL: {{^}}kernel_call_undef_func:
; GCN:    .amdhsa_next_free_vgpr 24
; GCN:    NumVgprs: 24
; GCN:    NumAgprs: 24
; GCN:    TotalNumVgprs: 24
; GCN:    VGPRBlocks: 5
; GCN:    NumVGPRsForWavesPerEU: 24
; GCN:    Occupancy: 10
define amdgpu_kernel void @kernel_call_undef_func() {
bb:
  call void @undef_func()
  ret void
}

attributes #0 = { nounwind noinline }
