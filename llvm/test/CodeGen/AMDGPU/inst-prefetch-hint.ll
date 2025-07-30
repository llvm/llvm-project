; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 --amdgpu-memcpy-loop-unroll=100000 < %s | FileCheck --check-prefixes=GCN,GFX11 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 --amdgpu-memcpy-loop-unroll=100000 < %s | FileCheck --check-prefixes=GCN,GFX12 %s

; GCN-LABEL: .amdhsa_kernel large
; GFX11: .amdhsa_inst_pref_size 3
; GFX11: codeLenInByte = 3{{[0-9][0-9]$}}
; GFX12: .amdhsa_inst_pref_size 4
; GFX12: codeLenInByte = 4{{[0-9][0-9]$}}
define amdgpu_kernel void @large(ptr addrspace(1) %out, ptr addrspace(1) %in) {
bb:
  call void @llvm.memcpy.p1.p3.i32(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 256, i1 false)
  ret void
}

; GCN-LABEL: .amdhsa_kernel small
; GCN: .amdhsa_inst_pref_size 1
; GCN: codeLenInByte = {{[0-9]$}}
define amdgpu_kernel void @small() {
bb:
  ret void
}

; Ignore inline asm in size calculation

; GCN-LABEL: .amdhsa_kernel inline_asm
; GCN: .amdhsa_inst_pref_size 1
; GCN: codeLenInByte = {{[0-9]$}}
define amdgpu_kernel void @inline_asm() {
bb:
  call void asm sideeffect ".fill 256, 4, 0", ""()
  ret void
}
