; RUN: llc -mtriple=amdgpu10.10-amd-amdhsa -mattr=+wavefrontsize32 < %s | FileCheck -check-prefixes=GCN,GFX10-32 %s
; RUN: llc -mtriple=amdgpu10.10-amd-amdhsa -mattr=+wavefrontsize64 < %s | FileCheck -check-prefixes=GCN,GFX10-64 %s
; RUN: llc -mtriple=amdgpu11.00-amd-amdhsa -mattr=+wavefrontsize32 < %s | FileCheck -check-prefixes=GCN,GFX10-32 %s
; RUN: llc -mtriple=amdgpu11.00-amd-amdhsa -mattr=+wavefrontsize64 < %s | FileCheck -check-prefixes=GCN,GFX10-64 %s

; GCN:      amdhsa.kernels:
; GCN:      .name: wavefrontsize
; GFX10-32: .wavefront_size: 32
; GFX10-64: .wavefront_size: 64
define amdgpu_kernel void @wavefrontsize() {
entry:
  ret void
}
