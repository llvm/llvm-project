; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -amdgpu-spill-sgpr-to-smem=1 -verify-machineinstrs < %s | FileCheck -check-prefix=TOSMEM -check-prefix=ALL %s

; FIXME: SGPR-to-SMEM requires an additional SGPR always to scavenge m0

; ALL-LABEL: {{^}}max_9_sgprs:
; ALL: SGPRBlocks: 1
; ALL: NumSGPRsForWavesPerEU: 9
define amdgpu_kernel void @max_9_sgprs() #0 {
  %one = load volatile i32, i32 addrspace(4)* undef
  %two = load volatile i32, i32 addrspace(4)* undef
  %three = load volatile i32, i32 addrspace(4)* undef
  %four = load volatile i32, i32 addrspace(4)* undef
  %five = load volatile i32, i32 addrspace(4)* undef
  %six = load volatile i32, i32 addrspace(4)* undef
  %seven = load volatile i32, i32 addrspace(4)* undef
  %eight = load volatile i32, i32 addrspace(4)* undef
  %nine = load volatile i32, i32 addrspace(4)* undef
  %ten = load volatile i32, i32 addrspace(4)* undef
  call void asm sideeffect "", "s,s,s,s,s,s,s,s"(i32 %one, i32 %two, i32 %three, i32 %four, i32 %five, i32 %six, i32 %seven, i32 %eight)
  store volatile i32 %one, i32 addrspace(1)* undef
  store volatile i32 %two, i32 addrspace(1)* undef
  store volatile i32 %three, i32 addrspace(1)* undef
  store volatile i32 %four, i32 addrspace(1)* undef
  store volatile i32 %five, i32 addrspace(1)* undef
  store volatile i32 %six, i32 addrspace(1)* undef
  store volatile i32 %seven, i32 addrspace(1)* undef
  store volatile i32 %eight, i32 addrspace(1)* undef
  store volatile i32 %nine, i32 addrspace(1)* undef
  store volatile i32 %ten, i32 addrspace(1)* undef
  ret void
}

attributes #0 = { nounwind "amdgpu-num-sgpr"="14" }
