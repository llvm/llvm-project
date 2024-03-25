; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -amdgpu-enable-vopd=0 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,FUNC %s
; RUN: llc -global-isel -mtriple=amdgcn -mcpu=gfx1100 -amdgpu-enable-vopd=0 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,FUNC %s
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx1200 -amdgpu-enable-vopd=0 -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=GFX12-ERR %s

; GFX12-ERR: LLVM ERROR: Cannot select: {{.*}} = DS_ORDERED_COUNT

; FUNC-LABEL: {{^}}ds_ordered_add:
; GCN-DAG: v_mov_b32_e32 v[[INCR:[0-9]+]], 31
; GCN-DAG: s_mov_b32 m0,
; GCN: ds_ordered_count v{{[0-9]+}}, v[[INCR]] offset:772 gds
define amdgpu_kernel void @ds_ordered_add(ptr addrspace(2) inreg %gds, ptr addrspace(1) %out) {
  %val = call i32@llvm.amdgcn.ds.ordered.add(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 false, i32 16777217, i1 true, i1 true)
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}ds_ordered_add_cs:
; GCN: v_mov_b32_e32 v[[INCR:[0-9]+]], 31
; GCN: s_mov_b32 m0, s0
; GCN-NEXT: ds_ordered_count v{{[0-9]+}}, v[[INCR]] offset:772 gds
; GCN-NEXT: s_waitcnt expcnt(0) lgkmcnt(0)
define amdgpu_cs float @ds_ordered_add_cs(ptr addrspace(2) inreg %gds) {
  %val = call i32@llvm.amdgcn.ds.ordered.add(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 false, i32 16777217, i1 true, i1 true)
  %r = bitcast i32 %val to float
  ret float %r
}

; FUNC-LABEL: {{^}}ds_ordered_add_ps:
; GCN: v_mov_b32_e32 v[[INCR:[0-9]+]], 31
; GCN: s_mov_b32 m0, s0
; GCN-NEXT: ds_ordered_count v{{[0-9]+}}, v[[INCR]] offset:772 gds
; GCN-NEXT: s_waitcnt expcnt(0) lgkmcnt(0)
define amdgpu_ps float @ds_ordered_add_ps(ptr addrspace(2) inreg %gds) {
  %val = call i32@llvm.amdgcn.ds.ordered.add(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 false, i32 16777217, i1 true, i1 true)
  %r = bitcast i32 %val to float
  ret float %r
}

; FUNC-LABEL: {{^}}ds_ordered_add_vs:
; GCN: v_mov_b32_e32 v[[INCR:[0-9]+]], 31
; GCN: s_mov_b32 m0, s0
; GCN-NEXT: ds_ordered_count v{{[0-9]+}}, v[[INCR]] offset:772 gds
; GCN-NEXT: s_waitcnt expcnt(0) lgkmcnt(0)
define amdgpu_vs float @ds_ordered_add_vs(ptr addrspace(2) inreg %gds) {
  %val = call i32@llvm.amdgcn.ds.ordered.add(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 false, i32 16777217, i1 true, i1 true)
  %r = bitcast i32 %val to float
  ret float %r
}

; FUNC-LABEL: {{^}}ds_ordered_add_gs:
; GCN: v_mov_b32_e32 v[[INCR:[0-9]+]], 31
; GCN: s_mov_b32 m0, s0
; GCN-NEXT: ds_ordered_count v{{[0-9]+}}, v[[INCR]] offset:772 gds
; GCN-NEXT: s_waitcnt expcnt(0) lgkmcnt(0)
define amdgpu_gs float @ds_ordered_add_gs(ptr addrspace(2) inreg %gds) {
  %val = call i32@llvm.amdgcn.ds.ordered.add(ptr addrspace(2) %gds, i32 31, i32 0, i32 0, i1 false, i32 16777217, i1 true, i1 true)
  %r = bitcast i32 %val to float
  ret float %r
}

declare i32 @llvm.amdgcn.ds.ordered.add(ptr addrspace(2) nocapture, i32, i32, i32, i1, i32, i1, i1)
