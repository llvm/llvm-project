; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx908 < %s | FileCheck -check-prefixes=CHECK,GFX908 %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx90a < %s | FileCheck -check-prefixes=CHECK,GFX90A %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx801 < %s | FileCheck -check-prefixes=CHECK,GFX801 %s

; COM: Comments for each kernel
; CHECK: kernel_32_agprs
; GFX908:   ; NumVgprs: 9
; GFX908    ; NumAgprs: 32
; GFX908    ; TotalNumVgprs: 32

; GFX90A:   ; NumVgprs: 9
; GFX90A    ; NumAgprs: 32
; GFX90A    ; TotalNumVgprs: 44

; GFX801:   ; NumVgprs: 9

; CHECK: kernel_40_vgprs
; GFX908:   ; NumVgprs: 40
; GFX908    ; NumAgprs: 16
; GFX908    ; TotalNumVgprs: 40

; GFX90A:   ; NumVgprs: 40
; GFX90A    ; NumAgprs: 16
; GFX90A    ; TotalNumVgprs: 56

; GFX801:   ; NumVgprs: 40

; COM: Metadata
; GFX908:    - .agpr_count:    32
; GFX908:      .vgpr_count:    32

; GFX90A:    - .agpr_count:    32
; GFX90A:      .vgpr_count:    44

; GFX801:      .vgpr_count:    9
define amdgpu_kernel void @kernel_32_agprs() #0 {
bb:
  call void asm sideeffect "", "~{v8}" ()
  call void asm sideeffect "", "~{a31}" ()
  ret void
}

; GFX908:    - .agpr_count:    16
; GFX908:      .vgpr_count:    40

; GFX90A:    - .agpr_count:    16
; GFX90A:      .vgpr_count:    56

; GFX801:      .vgpr_count:    40
define amdgpu_kernel void @kernel_40_vgprs() #0 {
bb:
  call void asm sideeffect "", "~{v39}" ()
  call void asm sideeffect "", "~{a15}" ()
  ret void
}

attributes #0 = { nounwind noinline "amdgpu-flat-work-group-size"="1,512" }
