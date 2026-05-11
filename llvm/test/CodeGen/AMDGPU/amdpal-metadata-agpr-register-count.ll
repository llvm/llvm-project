; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx90a < %s | FileCheck -check-prefixes=CHECK,GFX90A %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx908 < %s | FileCheck -check-prefixes=CHECK,GFX908 %s

; COM: Adapted from agpr-register-count.ll
; COM: GFX900 and below should not have .agpr_count present in the metadata


; CHECK:      .type          kernel_32_agprs
define amdgpu_kernel void @kernel_32_agprs() #0 {
bb:
  call void asm sideeffect "", "~{v8}" ()
  call void asm sideeffect "", "~{a31}" ()
  ret void
}

; CHECK:      .type          kernel_0_agprs
define amdgpu_kernel void @kernel_0_agprs() #0 {
bb:
  call void asm sideeffect "", "~{v0}" ()
  ret void
}

; CHECK:      .type           kernel_40_vgprs
define amdgpu_kernel void @kernel_40_vgprs() #0 {
bb:
  call void asm sideeffect "", "~{v39}" ()
  call void asm sideeffect "", "~{a15}" ()
  ret void
}

; CHECK:      .type          kernel_max_gprs
define amdgpu_kernel void @kernel_max_gprs() #0 {
bb:
  call void asm sideeffect "", "~{v255}" ()
  call void asm sideeffect "", "~{a255}" ()
  ret void
}

; CHECK:      .type          func_32_agprs
define void @func_32_agprs() #0 {
bb:
  call void asm sideeffect "", "~{v8}" ()
  call void asm sideeffect "", "~{a31}" ()
  ret void
}

; CHECK:      .type          kernel_call_func_32_agprs
define amdgpu_kernel void @kernel_call_func_32_agprs() #0 {
bb:
  call void @func_32_agprs() #0
  ret void
}

declare void @undef_func()

; CHECK:      .type          kernel_call_undef_func
; CHECK:      .set .Lkernel_call_undef_func.num_agpr, max(0, amdgpu.max_num_agpr)
define amdgpu_kernel void @kernel_call_undef_func() #0 {
bb:
  call void @undef_func()
  ret void
}

; CHECK: ; kernel_32_agprs Kernel info:
; CHECK:      NumAgprs:       32
; CHECK: ; kernel_0_agprs Kernel info:
; CHECK:      NumAgprs:       0
; CHECK: ; kernel_40_vgprs Kernel info:
; CHECK:      NumAgprs:       16
; CHECK: ; kernel_max_gprs Kernel info:
; CHECK:      NumAgprs:       256
; CHECK: ; func_32_agprs Function info:
; CHECK:      NumAgprs:       32
; CHECK: ; kernel_call_func_32_agprs Kernel info:
; CHECK:      NumAgprs:       32
; CHECK: ; kernel_call_undef_func Kernel info:
; CHECK:      NumAgprs:       32
; CHECK:      .set amdgpu.max_num_agpr, 32

; CHECK: ---
; CHECK:  amdpal.pipelines:
; GFX90A: agpr_count:  0x20
; GFX90A: vgpr_count:  0x40

; GFX908: agpr_count:  0x20
; GFX908: vgpr_count:  0x20

attributes #0 = { nounwind noinline "amdgpu-flat-work-group-size"="1,512" }
