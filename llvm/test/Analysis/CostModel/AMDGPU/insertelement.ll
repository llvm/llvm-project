; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa %s | FileCheck -check-prefixes=GCN,CI %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=fiji %s | FileCheck -check-prefixes=GCN,GFX89 %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 %s | FileCheck -check-prefixes=GCN,GFX89 %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa %s | FileCheck -check-prefixes=GCN,CI %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=fiji %s | FileCheck -check-prefixes=GCN,GFX89 %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 %s | FileCheck -check-prefixes=GCN,GFX89 %s

; GCN-LABEL: 'insertelement_v2i32'
; GCN: estimated cost of 0 for {{.*}} insertelement <2 x i32>
define amdgpu_kernel void @insertelement_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %vaddr) {
  %vec = load <2 x i32>, <2 x i32> addrspace(1)* %vaddr
  %insert = insertelement <2 x i32> %vec, i32 123, i32 1
  store <2 x i32> %insert, <2 x i32> addrspace(1)* %out
  ret void
}

; GCN-LABEL: 'insertelement_v2i64'
; GCN: estimated cost of 0 for {{.*}} insertelement <2 x i64>
define amdgpu_kernel void @insertelement_v2i64(<2 x i64> addrspace(1)* %out, <2 x i64> addrspace(1)* %vaddr) {
  %vec = load <2 x i64>, <2 x i64> addrspace(1)* %vaddr
  %insert = insertelement <2 x i64> %vec, i64 123, i64 1
  store <2 x i64> %insert, <2 x i64> addrspace(1)* %out
  ret void
}

; GCN-LABEL: 'insertelement_0_v2i16'
; CI: estimated cost of 1 for {{.*}} insertelement <2 x i16>
; GFX89: estimated cost of 0 for {{.*}} insertelement <2 x i16>
define amdgpu_kernel void @insertelement_0_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %vaddr) {
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %vaddr
  %insert = insertelement <2 x i16> %vec, i16 123, i16 0
  store <2 x i16> %insert, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: 'insertelement_1_v2i16'
; GCN: estimated cost of 1 for {{.*}} insertelement <2 x i16>
define amdgpu_kernel void @insertelement_1_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %vaddr) {
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %vaddr
  %insert = insertelement <2 x i16> %vec, i16 123, i16 1
  store <2 x i16> %insert, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: 'insertelement_1_v2i8'
; GCN: estimated cost of 1 for {{.*}} insertelement <2 x i8>
define amdgpu_kernel void @insertelement_1_v2i8(<2 x i8> addrspace(1)* %out, <2 x i8> addrspace(1)* %vaddr) {
  %vec = load <2 x i8>, <2 x i8> addrspace(1)* %vaddr
  %insert = insertelement <2 x i8> %vec, i8 123, i8 1
  store <2 x i8> %insert, <2 x i8> addrspace(1)* %out
  ret void
}
