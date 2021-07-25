; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX908 %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX90A %s

; GCN-LABEL: {{^}}kernel_32_agprs:
; GFX908: .amdhsa_next_free_vgpr 32
; GFX90A: .amdhsa_next_free_vgpr 44
; GFX90A: .amdhsa_accum_offset 12
; GCN:    NumVgprs: 9
; GCN:    NumAgprs: 32
; GFX908: TotalNumVgprs: 32
; GFX90A: TotalNumVgprs: 44
; GFX908: VGPRBlocks: 7
; GFX90A: VGPRBlocks: 5
; GFX908: NumVGPRsForWavesPerEU: 32
; GFX90A: NumVGPRsForWavesPerEU: 44
; GFX90A: AccumOffset: 12
; GCN:    Occupancy: 8
; GFX90A: COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 2
define amdgpu_kernel void @kernel_32_agprs() #0 {
bb:
  call void asm sideeffect "", "~{v8}" ()
  call void asm sideeffect "", "~{a31}" ()
  ret void
}

; GCN-LABEL: {{^}}kernel_0_agprs:
; GCN:    .amdhsa_next_free_vgpr 1
; GFX90A: .amdhsa_accum_offset 4
; GCN:    NumVgprs: 1
; GCN:    NumAgprs: 0
; GCN:    TotalNumVgprs: 1
; GCN:    VGPRBlocks: 0
; GCN:    NumVGPRsForWavesPerEU: 1
; GFX90A: AccumOffset: 4
; GFX908: Occupancy: 10
; GFX90A: Occupancy: 8
; GFX90A: COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 0
define amdgpu_kernel void @kernel_0_agprs() #0 {
bb:
  call void asm sideeffect "", "~{v0}" ()
  ret void
}

; GCN-LABEL: {{^}}kernel_40_vgprs:
; GFX908: .amdhsa_next_free_vgpr 40
; GFX90A: .amdhsa_next_free_vgpr 56
; GFX90A: .amdhsa_accum_offset 40
; GCN:    NumVgprs: 40
; GCN:    NumAgprs: 16
; GFX908: TotalNumVgprs: 40
; GFX90A: TotalNumVgprs: 56
; GFX908: VGPRBlocks: 9
; GFX90A: VGPRBlocks: 6
; GFX908: NumVGPRsForWavesPerEU: 40
; GFX90A: NumVGPRsForWavesPerEU: 56
; GFX90A: AccumOffset: 40
; GFX908: Occupancy: 6
; GFX90A: Occupancy: 8
; GFX90A: COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 9
define amdgpu_kernel void @kernel_40_vgprs() #0 {
bb:
  call void asm sideeffect "", "~{v39}" ()
  call void asm sideeffect "", "~{a15}" ()
  ret void
}

; GCN-LABEL: {{^}}func_32_agprs:
; GCN:    NumVgprs: 9
; GCN:    NumAgprs: 32
; GFX908: TotalNumVgprs: 32
; GFX90A: TotalNumVgprs: 44
define void @func_32_agprs() #0 {
bb:
  call void asm sideeffect "", "~{v8}" ()
  call void asm sideeffect "", "~{a31}" ()
  ret void
}

; GCN-LABEL: {{^}}func_32_vgprs:
; GCN:    NumVgprs: 32
; GCN:    NumAgprs: 9
; GFX908: TotalNumVgprs: 32
; GFX90A: TotalNumVgprs: 41
define void @func_32_vgprs() #0 {
bb:
  call void asm sideeffect "", "~{v31}" ()
  call void asm sideeffect "", "~{a8}" ()
  ret void
}

; GCN-LABEL: {{^}}func_0_agprs:
; GCN:    NumVgprs: 1
; GCN:    NumAgprs: 0
; GCN:    TotalNumVgprs: 1
define amdgpu_kernel void @func_0_agprs() #0 {
bb:
  call void asm sideeffect "", "~{v0}" ()
  ret void
}

; GCN-LABEL: {{^}}kernel_max_gprs:
; GFX908: .amdhsa_next_free_vgpr 256
; GFX90A: .amdhsa_next_free_vgpr 512
; GFX90A: .amdhsa_accum_offset 256
; GCN:    NumVgprs: 256
; GCN:    NumAgprs: 256
; GFX908: TotalNumVgprs: 256
; GFX90A: TotalNumVgprs: 512
; GFX908: VGPRBlocks: 63
; GFX90A: VGPRBlocks: 63
; GFX908: NumVGPRsForWavesPerEU: 256
; GFX90A: NumVGPRsForWavesPerEU: 512
; GFX90A: AccumOffset: 256
; GCN:    Occupancy: 1
; GFX90A: COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 63
define amdgpu_kernel void @kernel_max_gprs() #0 {
bb:
  call void asm sideeffect "", "~{v255}" ()
  call void asm sideeffect "", "~{a255}" ()
  ret void
}

; GCN-LABEL: {{^}}kernel_call_func_32_agprs:
; GFX908: .amdhsa_next_free_vgpr 32
; GFX90A: .amdhsa_accum_offset 12
; GCN:    NumVgprs: 9
; GCN:    NumAgprs: 32
; GFX908: TotalNumVgprs: 32
; GFX90A: TotalNumVgprs: 44
; GFX908: VGPRBlocks: 7
; GFX90A: VGPRBlocks: 5
; GFX908: NumVGPRsForWavesPerEU: 32
; GFX90A: NumVGPRsForWavesPerEU: 44
; GFX90A: AccumOffset: 12
; GCN:    Occupancy: 8
; GFX90A: COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 2
define amdgpu_kernel void @kernel_call_func_32_agprs() #0 {
bb:
  call void @func_32_agprs() #0
  ret void
}

; GCN-LABEL: {{^}}func_call_func_32_agprs:
; GCN:    NumVgprs: 9
; GCN:    NumAgprs: 32
; GFX908: TotalNumVgprs: 32
; GFX90A: TotalNumVgprs: 44
define void @func_call_func_32_agprs() #0 {
bb:
  call void @func_32_agprs() #0
  ret void
}

declare void @undef_func()

; GCN-LABEL: {{^}}kernel_call_undef_func:
; GFX908: .amdhsa_next_free_vgpr 32
; GFX90A: .amdhsa_next_free_vgpr 64
; GFX90A: .amdhsa_accum_offset 32
; GCN908: NumVgprs: 128
; GCN90A: NumVgprs: 256
; GCN:    NumAgprs: 32
; GFX908: TotalNumVgprs: 32
; GFX90A: TotalNumVgprs: 64
; GFX908: VGPRBlocks: 7
; GFX90A: VGPRBlocks: 7
; GFX908: NumVGPRsForWavesPerEU: 32
; GFX90A: NumVGPRsForWavesPerEU: 64
; GFX90A: AccumOffset: 32
; GFX908: Occupancy: 8
; GFX90A: Occupancy: 8
; GFX90A: COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 7
define amdgpu_kernel void @kernel_call_undef_func() #0 {
bb:
  call void @undef_func()
  ret void
}

attributes #0 = { nounwind noinline "amdgpu-flat-work-group-size"="1,512" }
