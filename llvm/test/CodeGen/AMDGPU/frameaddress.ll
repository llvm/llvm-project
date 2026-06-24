; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -global-isel -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck --check-prefix=GCN %s

; GCN-LABEL: {{^}}func1
; GCN: v_mov_b32_e32 v0, s33
; GCN: s_setpc_b64 s[30:31]
define ptr addrspace(5) @func1() nounwind {
entry:
  %0 = tail call ptr addrspace(5) @llvm.frameaddress.p5(i32 0)
  ret ptr addrspace(5) %0
}

; GCN-LABEL: {{^}}func2
; GCN: v_mov_b32_e32 v0, 0
; GCN: s_setpc_b64 s[30:31]
define ptr addrspace(5) @func2() nounwind {
entry:
  %0 = tail call ptr addrspace(5) @llvm.frameaddress.p5(i32 1)
  ret ptr addrspace(5) %0
}

; GCN-LABEL: {{^}}func3
; GCN: v_mov_b32_e32 v2, 0
; GCN: flat_store_dword v[{{[0-9]+:[0-9]+}}], v2
define amdgpu_kernel void @func3(ptr %out) nounwind {
entry:
  %tmp = tail call ptr addrspace(5) @llvm.frameaddress.p5(i32 0)
  store ptr addrspace(5) %tmp, ptr %out, align 4
  ret void
}

declare void @callee()

; GCN-LABEL: {{^}}multi_use:
; GCN: v_mov_b32_e32 v[[FP:[0-9]+]], s33
; GCN: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v[[FP]], off
; GCN: s_swappc_b64
; GCN: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v[[FP]], off
define void @multi_use() nounwind {
entry:
  %ret0 = tail call ptr addrspace(5) @llvm.frameaddress.p5(i32 0)
  store volatile ptr addrspace(5) %ret0, ptr addrspace(1) poison
  call void @callee()
  %ret1 = tail call ptr addrspace(5) @llvm.frameaddress.p5(i32 0)
  store volatile ptr addrspace(5) %ret1, ptr addrspace(1) poison
  ret void
}

declare ptr addrspace(5) @llvm.frameaddress.p5(i32) nounwind readnone
