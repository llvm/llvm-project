; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx908 < %s | FileCheck -check-prefixes=CHECK,GFX908 %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx90a < %s | FileCheck -check-prefixes=CHECK,GFX90A %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx801 < %s | FileCheck -check-prefixes=CHECK,GFX801 %s

define amdgpu_kernel void @kernel_32_agprs() #0 {
bb:
  call void asm sideeffect "", "~{v8}" ()
  call void asm sideeffect "", "~{a31}" ()
  ret void
}

define amdgpu_kernel void @kernel_40_vgprs() #0 {
bb:
  call void asm sideeffect "", "~{v39}" ()
  call void asm sideeffect "", "~{a15}" ()
  ret void
}

; CHECK: .section .AMDGPU.csdata
; CHECK: ; kernel_32_agprs Kernel info:
; GFX908:   ; NumVgprs: 9
; GFX908:   ; NumAgprs: 32
; GFX908:   ; TotalNumVgprs: 32

; GFX90A:   ; NumVgprs: 9
; GFX90A:   ; NumAgprs: 32
; GFX90A:   ; TotalNumVgprs: 44

; GFX801:   ; NumVgprs: 9

; CHECK: ; kernel_40_vgprs Kernel info:
; GFX908:   ; NumVgprs: 40
; GFX908:   ; NumAgprs: 16
; GFX908:   ; TotalNumVgprs: 40

; GFX90A:   ; NumVgprs: 40
; GFX90A:   ; NumAgprs: 16
; GFX90A:   ; TotalNumVgprs: 56

; GFX801:   ; NumVgprs: 40

; Metadata
; GFX908:    - .agpr_count:    32
; GFX908:      .vgpr_count:    32

; GFX90A:    - .agpr_count:    32
; GFX90A:      .vgpr_count:    44

; GFX801:      .vgpr_count:    9

; GFX908:    - .agpr_count:    16
; GFX908:      .vgpr_count:    40

; GFX90A:    - .agpr_count:    16
; GFX90A:      .vgpr_count:    56

; GFX801:      .vgpr_count:    40

attributes #0 = { nounwind noinline "amdgpu-flat-work-group-size"="1,512" }
