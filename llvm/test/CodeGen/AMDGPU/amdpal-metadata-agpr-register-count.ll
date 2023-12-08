; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx90a < %s | FileCheck -check-prefixes=CHECK,GFX90A %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx908 < %s | FileCheck -check-prefixes=CHECK,GFX908 %s

; COM: Adapted from agpr-register-count.ll
; COM: GFX900 and below should not have .agpr_count present in the metadata


; CHECK:      .type          kernel_32_agprs
; CHECK:      NumAgprs:       32
define amdgpu_kernel void @kernel_32_agprs() #0 {
bb:
  call void asm sideeffect "", "~{v8}" ()
  call void asm sideeffect "", "~{a31}" ()
  ret void
}

; CHECK:      .type          kernel_0_agprs
; CHECK:      NumAgprs:       0
define amdgpu_kernel void @kernel_0_agprs() #0 {
bb:
  call void asm sideeffect "", "~{v0}" ()
  ret void
}

; CHECK:      .type           kernel_40_vgprs
; CHECK:      NumAgprs:       16
define amdgpu_kernel void @kernel_40_vgprs() #0 {
bb:
  call void asm sideeffect "", "~{v39}" ()
  call void asm sideeffect "", "~{a15}" ()
  ret void
}

; CHECK:      .type          kernel_max_gprs
; CHECK:      NumAgprs:       256
define amdgpu_kernel void @kernel_max_gprs() #0 {
bb:
  call void asm sideeffect "", "~{v255}" ()
  call void asm sideeffect "", "~{a255}" ()
  ret void
}

; CHECK:      .type          func_32_agprs
; CHECK:      NumAgprs:       32
define void @func_32_agprs() #0 {
bb:
  call void asm sideeffect "", "~{v8}" ()
  call void asm sideeffect "", "~{a31}" ()
  ret void
}

; CHECK:      .type          kernel_call_func_32_agprs
; CHECK:      NumAgprs:       32
define amdgpu_kernel void @kernel_call_func_32_agprs() #0 {
bb:
  call void @func_32_agprs() #0
  ret void
}

declare void @undef_func()

; CHECK:      .type          kernel_call_undef_func
; CHECK:      NumAgprs:       32
define amdgpu_kernel void @kernel_call_undef_func() #0 {
bb:
  call void @undef_func()
  ret void
}

; CHECK: ---
; CHECK:  amdpal.pipelines:
; GFX90A: agpr_count:  0x20
; GFX90A: vgpr_count:  0x40

; GFX908: agpr_count:  0x20
; GFX908: vgpr_count:  0x20

attributes #0 = { nounwind noinline "amdgpu-flat-work-group-size"="1,512" }
