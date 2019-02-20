; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}cast_constant_i64_to_build_vector_v4i16:
; GCN: global_store_dwordx2
; GCN: global_store_dword v
; GCN: global_store_short
define amdgpu_kernel void @cast_constant_i64_to_build_vector_v4i16(i8 addrspace(1)* nocapture %data) {
entry:
  store i8 72, i8 addrspace(1)* %data, align 1
  %arrayidx1 = getelementptr inbounds i8, i8 addrspace(1)* %data, i64 1
  store i8 101, i8 addrspace(1)* %arrayidx1, align 1
  %arrayidx2 = getelementptr inbounds i8, i8 addrspace(1)* %data, i64 2
  store i8 108, i8 addrspace(1)* %arrayidx2, align 1
  %arrayidx3 = getelementptr inbounds i8, i8 addrspace(1)* %data, i64 3
  store i8 108, i8 addrspace(1)* %arrayidx3, align 1
  %arrayidx4 = getelementptr inbounds i8, i8 addrspace(1)* %data, i64 4
  store i8 111, i8 addrspace(1)* %arrayidx4, align 1
  %arrayidx5 = getelementptr inbounds i8, i8 addrspace(1)* %data, i64 5
  store i8 44, i8 addrspace(1)* %arrayidx5, align 1
  %arrayidx6 = getelementptr inbounds i8, i8 addrspace(1)* %data, i64 6
  store i8 32, i8 addrspace(1)* %arrayidx6, align 1
  %arrayidx7 = getelementptr inbounds i8, i8 addrspace(1)* %data, i64 7
  store i8 87, i8 addrspace(1)* %arrayidx7, align 1
  %arrayidx8 = getelementptr inbounds i8, i8 addrspace(1)* %data, i64 8
  store i8 111, i8 addrspace(1)* %arrayidx8, align 1
  %arrayidx9 = getelementptr inbounds i8, i8 addrspace(1)* %data, i64 9
  store i8 114, i8 addrspace(1)* %arrayidx9, align 1
  %arrayidx10 = getelementptr inbounds i8, i8 addrspace(1)* %data, i64 10
  store i8 108, i8 addrspace(1)* %arrayidx10, align 1
  %arrayidx11 = getelementptr inbounds i8, i8 addrspace(1)* %data, i64 11
  store i8 100, i8 addrspace(1)* %arrayidx11, align 1
  %arrayidx12 = getelementptr inbounds i8, i8 addrspace(1)* %data, i64 12
  store i8 33, i8 addrspace(1)* %arrayidx12, align 1
  %arrayidx13 = getelementptr inbounds i8, i8 addrspace(1)* %data, i64 13
  store i8 72, i8 addrspace(1)* %arrayidx13, align 1
  ret void
}

