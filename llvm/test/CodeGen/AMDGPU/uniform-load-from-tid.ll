; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,W32 --enable-var-scope %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize64 -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,W64 --enable-var-scope %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -S -amdgpu-annotate-uniform < %s | FileCheck --check-prefixes=OPT,OPT-W32 --enable-var-scope %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=+wavefrontsize64 -S -amdgpu-annotate-uniform < %s | FileCheck --check-prefixes=OPT,OPT-W64 --enable-var-scope %s

; GCN-LABEL: {{^}}lshr_threadid:
; W64: global_load_dword
; W32: v_readfirstlane_b32 [[OFFSET:s[0-9]+]], v0
; W32: s_load_dword s{{[0-9]+}}, s[{{[0-9:]+}}], [[OFFSET]]

; OPT-LABEL: @lshr_threadid
; OPT-W64: %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %div4{{$}}
; OPT-W32: %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %div4, !amdgpu.uniform !0
define amdgpu_kernel void @lshr_threadid(ptr addrspace(1) align 4 %in, ptr addrspace(1) align 4 %out)  {
entry:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %div = lshr i32 %lid, 5
  %div4 = zext i32 %div to i64
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %div4
  %load = load i32, ptr addrspace(1) %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %div4
  store i32 %load, ptr addrspace(1) %arrayidx2, align 4
  ret void
}

; GCN-LABEL: {{^}}ashr_threadid:
; W64: global_load_dword
; W32: v_readfirstlane_b32 [[OFFSET:s[0-9]+]], v0
; W32: s_load_dword s{{[0-9]+}}, s[{{[0-9:]+}}], [[OFFSET]]

; OPT-LABEL: @ashr_threadid
; OPT-W64: %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %div4{{$}}
; OPT-W32: %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %div4, !amdgpu.uniform !0
define amdgpu_kernel void @ashr_threadid(ptr addrspace(1) align 4 %in, ptr addrspace(1) align 4 %out)  {
entry:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %div = ashr i32 %lid, 5
  %div4 = zext i32 %div to i64
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %div4
  %load = load i32, ptr addrspace(1) %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %div4
  store i32 %load, ptr addrspace(1) %arrayidx2, align 4
  ret void
}

; GCN-LABEL: {{^}}and_threadid:
; W64: global_load_dword
; W32: v_readfirstlane_b32 [[OFFSET:s[0-9]+]], v0
; W32: s_load_dword s{{[0-9]+}}, s[{{[0-9:]+}}], [[OFFSET]]

; OPT-LABEL: @and_threadid
; OPT-W64: %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %div4{{$}}
; OPT-W32: %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %div4, !amdgpu.uniform !0
define amdgpu_kernel void @and_threadid(ptr addrspace(1) align 4 %in, ptr addrspace(1) align 4 %out)  {
entry:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %and = and i32 %lid, -32
  %div4 = zext i32 %and to i64
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %div4
  %load = load i32, ptr addrspace(1) %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %div4
  store i32 %load, ptr addrspace(1) %arrayidx2, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
