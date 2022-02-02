; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck -check-prefixes=CHECK,GFX90A %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck -check-prefixes=CHECK,GFX908 %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx801 -verify-machineinstrs < %s | FileCheck -check-prefixes=CHECK,GFX801 %s

; COM: Adapted from agpr-register-count.ll
; COM: GFX900 and below should not have .agpr_count present in the metadata

; CHECK: ---
; CHECK:  amdhsa.kernels:

; GFX90A:    - .agpr_count:    32
; GFX908:    - .agpr_count:    32
; GFX801-NOT:    - .agpr_count:
; CHECK:      .name:          kernel_32_agprs
; GFX90A:      .vgpr_count:    44
; GFX908:      .vgpr_count:    32
; GFX801:      .vgpr_count:    9
define amdgpu_kernel void @kernel_32_agprs() #0 {
bb:
  call void asm sideeffect "", "~{v8}" ()
  call void asm sideeffect "", "~{a31}" ()
  ret void
}

; GFX90A:    - .agpr_count:    0
; GFX908:    - .agpr_count:    0
; GFX801-NOT:    - .agpr_count:
; CHECK:      .name:          kernel_0_agprs
; GFX90A:      .vgpr_count:    1
; GFX908:      .vgpr_count:    1
; GFX801:      .vgpr_count:    1
define amdgpu_kernel void @kernel_0_agprs() #0 {
bb:
  call void asm sideeffect "", "~{v0}" ()
  ret void
}

; GFX90A:    - .agpr_count:    16
; GFX908:    - .agpr_count:    16
; GFX801-NOT:    - .agpr_count:
; CHECK:      .name:          kernel_40_vgprs
; GFX90A:      .vgpr_count:    56
; GFX908:      .vgpr_count:    40
; GFX801:      .vgpr_count:    40
define amdgpu_kernel void @kernel_40_vgprs() #0 {
bb:
  call void asm sideeffect "", "~{v39}" ()
  call void asm sideeffect "", "~{a15}" ()
  ret void
}

; GFX90A:    - .agpr_count:    256
; GFX908:    - .agpr_count:    256
; GFX801-NOT:    - .agpr_count:
; CHECK:      .name:          kernel_max_gprs
; GFX90A:      .vgpr_count:    512
; GFX908:      .vgpr_count:    256
; GFX801:      .vgpr_count:    256
define amdgpu_kernel void @kernel_max_gprs() #0 {
bb:
  call void asm sideeffect "", "~{v255}" ()
  call void asm sideeffect "", "~{a255}" ()
  ret void
}

define void @func_32_agprs() #0 {
bb:
  call void asm sideeffect "", "~{v8}" ()
  call void asm sideeffect "", "~{a31}" ()
  ret void
}

; GFX90A:    - .agpr_count:    32
; GFX908:    - .agpr_count:    32
; GFX801-NOT:    - .agpr_count:
; CHECK:      .name:          kernel_call_func_32_agprs
; GFX90A:      .vgpr_count:    44
; GFX908:      .vgpr_count:    32
; GFX801:      .vgpr_count:    9
define amdgpu_kernel void @kernel_call_func_32_agprs() #0 {
bb:
  call void @func_32_agprs() #0
  ret void
}

declare void @undef_func()

; GFX90A:    - .agpr_count:    32
; GFX908:    - .agpr_count:    32
; GFX801-NOT:    - .agpr_count:
; CHECK:      .name:          kernel_call_undef_func
; GFX90A:      .vgpr_count:    64
; GFX908:      .vgpr_count:    32
; GFX801:      .vgpr_count:    32
define amdgpu_kernel void @kernel_call_undef_func() #0 {
bb:
  call void @undef_func()
  ret void
}

attributes #0 = { nounwind noinline "amdgpu-flat-work-group-size"="1,512" }
