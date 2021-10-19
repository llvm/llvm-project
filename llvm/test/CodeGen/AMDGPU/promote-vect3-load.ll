; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s

; The type promotion for the vector loads v3i32/v3f32 into v4i32/v4f32 is enabled
; only when the alignment is 8-byte or higher.
; Otherwise, split the load into two separate loads (dwordx2 + dword).
; This type promotion on smaller aligned loads can cause a page fault error
; while accessing one extra dword beyond the buffer.

define protected amdgpu_kernel void @load_v3i32_align4(<3 x i32> addrspace(1)* %arg) #0 {
; GCN-LABEL: load_v3i32_align4:
; GCN:       ; %bb.0:
; GCN:         s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_load_dwordx2 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0x0
; GCN-NEXT:    s_load_dword s{{[0-9]+}}, s[0:1], 0x8
  %vec = load <3 x i32>, <3 x i32> addrspace(1)* %arg, align 4
  store <3 x i32> %vec, <3 x i32> addrspace(1)* undef, align 4
  ret void
}

define protected amdgpu_kernel void @load_v3i32_align8(<3 x i32> addrspace(1)* %arg) #0 {
; GCN-LABEL: load_v3i32_align8:
; GCN:       ; %bb.0:
; GCN:         s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_load_dwordx4 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0x0
  %vec = load <3 x i32>, <3 x i32> addrspace(1)* %arg, align 8
  store <3 x i32> %vec, <3 x i32> addrspace(1)* undef, align 8
  ret void
}

define protected amdgpu_kernel void @load_v3i32_align16(<3 x i32> addrspace(1)* %arg) #0 {
; GCN-LABEL: load_v3i32_align16:
; GCN:       ; %bb.0:
; GCN:         s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_load_dwordx4 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0x0
  %vec = load <3 x i32>, <3 x i32> addrspace(1)* %arg, align 16
  store <3 x i32> %vec, <3 x i32> addrspace(1)* undef, align 16
  ret void
}

define protected amdgpu_kernel void @load_v3f32_align4(<3 x float> addrspace(1)* %arg) #0 {
; GCN-LABEL: load_v3f32_align4:
; GCN:       ; %bb.0:
; GCN:         s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_load_dwordx2 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0x0
; GCN-NEXT:    s_load_dword s{{[0-9]+}}, s[0:1], 0x8
  %vec = load <3 x float>, <3 x float> addrspace(1)* %arg, align 4
  store <3 x float> %vec, <3 x float> addrspace(1)* undef, align 4
  ret void
}

define protected amdgpu_kernel void @load_v3f32_align8(<3 x float> addrspace(1)* %arg) #0 {
; GCN-LABEL: load_v3f32_align8:
; GCN:       ; %bb.0:
; GCN:         s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_load_dwordx4 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0x0
  %vec = load <3 x float>, <3 x float> addrspace(1)* %arg, align 8
  store <3 x float> %vec, <3 x float> addrspace(1)* undef, align 8
  ret void
}

define protected amdgpu_kernel void @load_v3f32_align16(<3 x float> addrspace(1)* %arg) #0 {
; GCN-LABEL: load_v3f32_align16:
; GCN:       ; %bb.0:
; GCN:         s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_load_dwordx4 s{{\[[0-9]:[0-9]+\]}}, s[0:1], 0x0
  %vec = load <3 x float>, <3 x float> addrspace(1)* %arg, align 16
  store <3 x float> %vec, <3 x float> addrspace(1)* undef, align 16
  ret void
}

attributes #0 = { nounwind noinline }
