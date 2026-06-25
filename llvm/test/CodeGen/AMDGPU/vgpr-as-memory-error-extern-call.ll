; RUN: not llc -mtriple=amdgcn -mcpu=gfx942 < %s 2>&1 | FileCheck %s

; The "VGPR as memory" file lives in low, caller-saved VGPRs that only call-graph
; members reserve. A call to an external (or indirect) callee would clobber it,
; so AMDGPULowerModuleVGPRs diagnoses it at the IR level, and the post-RA
; AMDGPUPrivateObjectVGPRs pass independently diagnoses the (attribute-less)
; machine call - this also covers calls introduced after the module pass.

@g = internal addrspace(13) global i32 poison

declare void @ext()

; CHECK: error: {{.*}}'VGPR as memory' is not supported in a function that makes an indirect call or a call outside its call graph
; CHECK: error: {{.*}}call to a function that clobbers the 'VGPR as memory' reserved file
define amdgpu_kernel void @extern_call() {
  store i32 1, ptr addrspace(13) @g
  call void @ext()
  ret void
}
