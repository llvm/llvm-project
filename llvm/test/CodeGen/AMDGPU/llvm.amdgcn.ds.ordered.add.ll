; RUN: llc -global-isel=0 -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,FUNC %s
; RUN: llc -global-isel=1 -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,FUNC %s
; RUN: llc -global-isel=0 -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,FUNC %s
; RUN: llc -global-isel=1 -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,FUNC %s
; RUN: llc -global-isel=0 -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VIGFX9,FUNC %s
; RUN: llc -global-isel=1 -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VIGFX9,FUNC %s
; RUN: llc -global-isel=0 -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VIGFX9,FUNC %s
; RUN: llc -global-isel=1 -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VIGFX9,FUNC %s

; FUNC-LABEL: {{^}}ds_ordered_add:
; GCN-DAG: v_mov_b32_e32 v[[INCR:[0-9]+]], 31
; GCN-DAG: s_mov_b32 m0,
; GCN: ds_ordered_count v{{[0-9]+}}, v[[INCR]] offset:772 gds
define amdgpu_kernel void @ds_ordered_add(i32 addrspace(2)* inreg %gds, i32 addrspace(1)* %out) {
  %val = call i32@llvm.amdgcn.ds.ordered.add(i32 addrspace(2)* %gds, i32 31, i32 0, i32 0, i1 false, i32 1, i1 true, i1 true)
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; Below are various modifications of input operands and shader types.

; FUNC-LABEL: {{^}}ds_ordered_add_counter2:
; GCN-DAG: v_mov_b32_e32 v[[INCR:[0-9]+]], 31
; GCN-DAG: s_mov_b32 m0,
; GCN: ds_ordered_count v{{[0-9]+}}, v[[INCR]] offset:776 gds
define amdgpu_kernel void @ds_ordered_add_counter2(i32 addrspace(2)* inreg %gds, i32 addrspace(1)* %out) {
  %val = call i32@llvm.amdgcn.ds.ordered.add(i32 addrspace(2)* %gds, i32 31, i32 0, i32 0, i1 false, i32 2, i1 true, i1 true)
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}ds_ordered_add_nodone:
; GCN-DAG: v_mov_b32_e32 v[[INCR:[0-9]+]], 31
; GCN-DAG: s_mov_b32 m0,
; GCN: ds_ordered_count v{{[0-9]+}}, v[[INCR]] offset:260 gds
define amdgpu_kernel void @ds_ordered_add_nodone(i32 addrspace(2)* inreg %gds, i32 addrspace(1)* %out) {
  %val = call i32@llvm.amdgcn.ds.ordered.add(i32 addrspace(2)* %gds, i32 31, i32 0, i32 0, i1 false, i32 1, i1 true, i1 false)
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}ds_ordered_add_norelease:
; GCN-DAG: v_mov_b32_e32 v[[INCR:[0-9]+]], 31
; GCN-DAG: s_mov_b32 m0,
; GCN: ds_ordered_count v{{[0-9]+}}, v[[INCR]] offset:4 gds
define amdgpu_kernel void @ds_ordered_add_norelease(i32 addrspace(2)* inreg %gds, i32 addrspace(1)* %out) {
  %val = call i32@llvm.amdgcn.ds.ordered.add(i32 addrspace(2)* %gds, i32 31, i32 0, i32 0, i1 false, i32 1, i1 false, i1 false)
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}ds_ordered_add_cs:
; GCN: v_mov_b32_e32 v[[INCR:[0-9]+]], 31
; GCN: s_mov_b32 m0, s0
; VIGFX9-NEXT: s_nop 0
; GCN-NEXT: ds_ordered_count v{{[0-9]+}}, v[[INCR]] offset:772 gds
; GCN-NEXT: s_waitcnt expcnt(0) lgkmcnt(0)
define amdgpu_cs float @ds_ordered_add_cs(i32 addrspace(2)* inreg %gds) {
  %val = call i32@llvm.amdgcn.ds.ordered.add(i32 addrspace(2)* %gds, i32 31, i32 0, i32 0, i1 false, i32 1, i1 true, i1 true)
  %r = bitcast i32 %val to float
  ret float %r
}

; FUNC-LABEL: {{^}}ds_ordered_add_default_cc:
; GCN: v_mov_b32_e32 v[[INCR:[0-9]+]], 31
; GCN: s_mov_b32 m0, 0{{$}}
; VIGFX9-NEXT: s_nop 0
; GCN-NEXT: ds_ordered_count v{{[0-9]+}}, v[[INCR]] offset:772 gds
; GCN-NEXT: s_waitcnt expcnt(0) lgkmcnt(0)
define float @ds_ordered_add_default_cc() {
  %val = call i32 @llvm.amdgcn.ds.ordered.add(i32 addrspace(2)* null, i32 31, i32 0, i32 0, i1 false, i32 1, i1 true, i1 true)
  %r = bitcast i32 %val to float
  ret float %r
}

; FUNC-LABEL: {{^}}ds_ordered_add_fastcc:
; GCN: v_mov_b32_e32 v[[INCR:[0-9]+]], 31
; GCN: s_mov_b32 m0, 0{{$}}
; VIGFX9-NEXT: s_nop 0
; GCN-NEXT: ds_ordered_count v{{[0-9]+}}, v[[INCR]] offset:772 gds
; GCN-NEXT: s_waitcnt expcnt(0) lgkmcnt(0)
define fastcc float @ds_ordered_add_fastcc() {
  %val = call i32 @llvm.amdgcn.ds.ordered.add(i32 addrspace(2)* null, i32 31, i32 0, i32 0, i1 false, i32 1, i1 true, i1 true)
  %r = bitcast i32 %val to float
  ret float %r
}

; FUNC-LABEL: {{^}}ds_ordered_add_func:
; GCN: v_mov_b32_e32 v[[INCR:[0-9]+]], 31
; GCN: s_mov_b32 m0, 0{{$}}
; VIGFX9-NEXT: s_nop 0
; GCN-NEXT: ds_ordered_count v{{[0-9]+}}, v[[INCR]] offset:772 gds
; GCN-NEXT: s_waitcnt expcnt(0) lgkmcnt(0)
define float @ds_ordered_add_func() {
  %val = call i32@llvm.amdgcn.ds.ordered.add(i32 addrspace(2)* null, i32 31, i32 0, i32 0, i1 false, i32 1, i1 true, i1 true)
  %r = bitcast i32 %val to float
  ret float %r
}

; FUNC-LABEL: {{^}}ds_ordered_add_ps:
; GCN: v_mov_b32_e32 v[[INCR:[0-9]+]], 31
; GCN: s_mov_b32 m0, s0
; VIGFX9-NEXT: s_nop 0
; GCN-NEXT: ds_ordered_count v{{[0-9]+}}, v[[INCR]] offset:1796 gds
; GCN-NEXT: s_waitcnt expcnt(0) lgkmcnt(0)
define amdgpu_ps float @ds_ordered_add_ps(i32 addrspace(2)* inreg %gds) {
  %val = call i32@llvm.amdgcn.ds.ordered.add(i32 addrspace(2)* %gds, i32 31, i32 0, i32 0, i1 false, i32 1, i1 true, i1 true)
  %r = bitcast i32 %val to float
  ret float %r
}

; FUNC-LABEL: {{^}}ds_ordered_add_vs:
; GCN: v_mov_b32_e32 v[[INCR:[0-9]+]], 31
; GCN: s_mov_b32 m0, s0
; VIGFX9-NEXT: s_nop 0
; GCN-NEXT: ds_ordered_count v{{[0-9]+}}, v[[INCR]] offset:2820 gds
; GCN-NEXT: s_waitcnt expcnt(0) lgkmcnt(0)
define amdgpu_vs float @ds_ordered_add_vs(i32 addrspace(2)* inreg %gds) {
  %val = call i32@llvm.amdgcn.ds.ordered.add(i32 addrspace(2)* %gds, i32 31, i32 0, i32 0, i1 false, i32 1, i1 true, i1 true)
  %r = bitcast i32 %val to float
  ret float %r
}

; FUNC-LABEL: {{^}}ds_ordered_add_gs:
; GCN: v_mov_b32_e32 v[[INCR:[0-9]+]], 31
; GCN: s_mov_b32 m0, s0
; VIGFX9-NEXT: s_nop 0
; GCN-NEXT: ds_ordered_count v{{[0-9]+}}, v[[INCR]] offset:3844 gds
; GCN-NEXT: s_waitcnt expcnt(0) lgkmcnt(0)
define amdgpu_gs float @ds_ordered_add_gs(i32 addrspace(2)* inreg %gds) {
  %val = call i32@llvm.amdgcn.ds.ordered.add(i32 addrspace(2)* %gds, i32 31, i32 0, i32 0, i1 false, i32 1, i1 true, i1 true)
  %r = bitcast i32 %val to float
  ret float %r
}

declare i32 @llvm.amdgcn.ds.ordered.add(i32 addrspace(2)* nocapture, i32, i32, i32, i1, i32, i1, i1)
