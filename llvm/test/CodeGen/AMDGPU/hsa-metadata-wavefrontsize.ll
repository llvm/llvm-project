; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 < %s | FileCheck -check-prefixes=GCN,GFX10-32 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 < %s | FileCheck -check-prefixes=GCN,GFX10-64 %s

; GCN:      amdhsa.kernels:
; GCN:      .name: wavefrontsize
; GFX10-32: .wavefront_size: 32
; GFX10-64: .wavefront_size: 64
define amdgpu_kernel void @wavefrontsize() {
entry:
  ret void
}
