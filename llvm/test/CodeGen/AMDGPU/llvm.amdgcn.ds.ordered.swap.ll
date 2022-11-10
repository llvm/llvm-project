; RUN: llc -global-isel=0 -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,FUNC %s
; RUN: llc -global-isel=1 -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,FUNC %s
; RUN: llc -global-isel=0 -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,FUNC %s
; RUN: llc -global-isel=1 -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,FUNC %s
; RUN: llc -global-isel=0 -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VIGFX9,FUNC %s
; RUN: llc -global-isel=1 -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VIGFX9,FUNC %s
; RUN: llc -global-isel=0 -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VIGFX9,FUNC %s
; RUN: llc -global-isel=1 -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VIGFX9,FUNC %s

; FUNC-LABEL: {{^}}ds_ordered_swap:
; GCN: s_mov_b32 m0, s0
; VIGFX9-NEXT: s_nop 0
; GCN-NEXT: ds_ordered_count v{{[0-9]+}}, v0 offset:4868 gds
; GCN-NEXT: s_waitcnt expcnt(0) lgkmcnt(0)
define amdgpu_cs float @ds_ordered_swap(i32 addrspace(2)* inreg %gds, i32 %value) {
  %val = call i32@llvm.amdgcn.ds.ordered.swap(i32 addrspace(2)* %gds, i32 %value, i32 0, i32 0, i1 false, i32 1, i1 true, i1 true)
  %r = bitcast i32 %val to float
  ret float %r
}

; FUNC-LABEL: {{^}}ds_ordered_swap_conditional:
; GCN: v_cmp_ne_u32_e32 vcc, 0, v[[VALUE:[0-9]+]]
; GCN: s_and_saveexec_b64 s[[SAVED:\[[0-9]+:[0-9]+\]]], vcc
; // We have to use s_cbranch, because ds_ordered_count has side effects with EXEC=0
; GCN: s_cbranch_execz [[BB:.LBB._.]]
; GCN: s_mov_b32 m0, s0
; VIGFX9-NEXT: s_nop 0
; GCN-NEXT: ds_ordered_count v{{[0-9]+}}, v[[VALUE]] offset:4868 gds
; GCN-NEXT: [[BB]]:
; // Wait for expcnt(0) before modifying EXEC
; GCN-NEXT: s_waitcnt expcnt(0)
; GCN-NEXT: s_or_b64 exec, exec, s[[SAVED]]
; GCN-NEXT: s_waitcnt lgkmcnt(0)
define amdgpu_cs float @ds_ordered_swap_conditional(i32 addrspace(2)* inreg %gds, i32 %value) {
entry:
  %c = icmp ne i32 %value, 0
  br i1 %c, label %if-true, label %endif

if-true:
  %val = call i32@llvm.amdgcn.ds.ordered.swap(i32 addrspace(2)* %gds, i32 %value, i32 0, i32 0, i1 false, i32 1, i1 true, i1 true)
  br label %endif

endif:
  %v = phi i32 [ %val, %if-true ], [ undef, %entry ]
  %r = bitcast i32 %v to float
  ret float %r
}

declare i32 @llvm.amdgcn.ds.ordered.swap(i32 addrspace(2)* nocapture, i32, i32, i32, i1, i32, i1, i1)
