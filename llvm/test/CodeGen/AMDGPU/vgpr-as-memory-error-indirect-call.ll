; RUN: not llc -mtriple=amdgcn -mcpu=gfx942 < %s 2>&1 | FileCheck %s

; An indirect call cannot be proven to stay within the call graph that reserves
; the "VGPR as memory" file, so it could clobber the file. AMDGPULowerModuleVGPRs
; diagnoses it (the callee is unknown at the IR level).

@g = internal addrspace(13) global i32 poison

; CHECK: error: {{.*}}'VGPR as memory' is not supported in a function that makes an indirect call or a call outside its call graph
; CHECK: error: {{.*}}call to a function that clobbers the 'VGPR as memory' reserved file
define amdgpu_kernel void @indirect_call(ptr %fp) {
  store i32 1, ptr addrspace(13) @g
  call void %fp()
  ret void
}
