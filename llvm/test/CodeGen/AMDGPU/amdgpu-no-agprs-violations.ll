; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 < %s | FileCheck -check-prefixes=CHECK,GFX908 %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a < %s 2> %t.err | FileCheck -check-prefixes=CHECK,GFX90A %s
; RUN: FileCheck -check-prefix=ERR < %t.err %s

; Test undefined behavior where a function ends up needing AGPRs that
; was marked with "amdgpu-num-agpr="="0". There should be no asserts.

; TODO: Should this be an error, or let UB happen?

; ERR: error: <unknown>:0:0: no registers from class available to allocate in function 'kernel_illegal_agpr_use_asm'
; ERR: error: <unknown>:0:0: no registers from class available to allocate in function 'func_illegal_agpr_use_asm'
; ERR: error: <unknown>:0:0: no registers from class available to allocate in function 'kernel_calls_mfma.f32.32x32x1f32'

; CHECK: {{^}}kernel_illegal_agpr_use_asm:
; CHECK: ; use a0

; CHECK: NumVgprs: 0
; CHECK: NumAgprs: 1
define amdgpu_kernel void @kernel_illegal_agpr_use_asm() #0 {
  call void asm sideeffect "; use $0", "a"(i32 poison)
  ret void
}

; CHECK: {{^}}func_illegal_agpr_use_asm:
; CHECK: ; use a0

; CHECK: NumVgprs: 0
; CHECK: NumAgprs: 1
define void @func_illegal_agpr_use_asm() #0 {
  call void asm sideeffect "; use $0", "a"(i32 poison)
  ret void
}

; CHECK-LABEL: {{^}}kernel_calls_mfma.f32.32x32x1f32:
; CHECK: v_accvgpr_write_b32
; CHECK: v_accvgpr_read_b32

; GFX908: NumVgprs: 5
; GFX90A: NumVgprs: 36
; CHECK: NumAgprs: 32

; GFX908: TotalNumVgprs: 32
; GFX90A: TotalNumVgprs: 68
define amdgpu_kernel void @kernel_calls_mfma.f32.32x32x1f32(ptr addrspace(1) %out, float %a, float %b, <32 x float> %c) #0 {
  %result = call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float %a, float %b, <32 x float> %c, i32 0, i32 0, i32 0)
  store <32 x float> %result, ptr addrspace(1) %out
  ret void
}

attributes #0 = { "amdgpu-num-agpr"="0" }
