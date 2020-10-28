; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 --amdhsa-code-object-version=2 -mattr=+wavefrontsize32,-wavefrontsize64 < %s | FileCheck -check-prefixes=GCN,GFX10-32 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 --amdhsa-code-object-version=2 -mattr=-wavefrontsize32,+wavefrontsize64 < %s | FileCheck -check-prefixes=GCN,GFX10-64 %s

; GCN:      ---
; GCN:      Kernels:
; GCN:        - Name: wavefrontsize
; GCN:          CodeProps:
; GFX10-32:       WavefrontSize: 32
; GFX10-64:       WavefrontSize: 64
; GCN:      ...
define amdgpu_kernel void @wavefrontsize() {
entry:
  ret void
}
