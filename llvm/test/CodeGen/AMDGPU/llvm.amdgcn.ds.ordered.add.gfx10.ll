; RUN: llc -global-isel=0 -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s
; RUN: llc -global-isel=1 -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s
; RUN: llc -global-isel=0 -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s
; RUN: llc -global-isel=1 -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s

; GCN-LABEL: {{^}}ds_ordered_add:
; GCN-DAG: v_{{(dual_)?}}mov_b32{{(_e32)?}} v[[INCR:[0-9]+]], 31
; GCN-DAG: s_mov_b32 m0,
; GCN: ds_ordered_count v{{[0-9]+}}, v[[INCR]] offset:772 gds
define amdgpu_kernel void @ds_ordered_add(i32 addrspace(2)* inreg %gds, i32 addrspace(1)* %out) {
  %val = call i32@llvm.amdgcn.ds.ordered.add(i32 addrspace(2)* %gds, i32 31, i32 0, i32 0, i1 false, i32 16777217, i1 true, i1 true)
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}ds_ordered_add_4dw:
; GCN-DAG: v_{{(dual_)?}}mov_b32{{(_e32)?}} v[[INCR:[0-9]+]], 31
; GCN-DAG: s_mov_b32 m0,
; GCN: ds_ordered_count v{{[0-9]+}}, v[[INCR]] offset:49924 gds
define amdgpu_kernel void @ds_ordered_add_4dw(i32 addrspace(2)* inreg %gds, i32 addrspace(1)* %out) {
  %val = call i32@llvm.amdgcn.ds.ordered.add(i32 addrspace(2)* %gds, i32 31, i32 0, i32 0, i1 false, i32 67108865, i1 true, i1 true)
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.ds.ordered.add(i32 addrspace(2)* nocapture, i32, i32, i32, i1, i32, i1, i1)
