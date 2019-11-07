; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 %s | FileCheck -check-prefixes=GFX9,GCN,TPT %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=fiji %s | FileCheck -check-prefixes=VI,GCN,TPT %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx900 %s | FileCheck -check-prefixes=GFX9,GCN,CS %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa -mcpu=fiji %s | FileCheck -check-prefixes=VI,GCN,CS %s

; GCN-LABEL: 'shufflevector_00_v2i16'
; GFX9: estimated cost of 0 for {{.*}} shufflevector <2 x i16> %vec, <2 x i16> undef, <2 x i32> zeroinitializer
; VI: estimated cost of 1 for {{.*}} shufflevector <2 x i16> %vec, <2 x i16> undef, <2 x i32> zeroinitializer
define amdgpu_kernel void @shufflevector_00_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %vaddr) {
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %vaddr
  %shuf = shufflevector <2 x i16> %vec, <2 x i16> undef, <2 x i32> zeroinitializer
  store <2 x i16> %shuf, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: 'shufflevector_01_v2i16'
; GCN: estimated cost of 0 for {{.*}} shufflevector <2 x i16> %vec, <2 x i16> undef, <2 x i32> <i32 0, i32 1>
define amdgpu_kernel void @shufflevector_01_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %vaddr) {
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %vaddr
  %shuf = shufflevector <2 x i16> %vec, <2 x i16> undef, <2 x i32> <i32 0, i32 1>
  store <2 x i16> %shuf, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: 'shufflevector_10_v2i16'
; GFX9: estimated cost of 0 for {{.*}} shufflevector <2 x i16> %vec, <2 x i16> undef, <2 x i32> <i32 1, i32 0>
; VI: estimated cost of 2 for {{.*}} shufflevector <2 x i16> %vec, <2 x i16> undef, <2 x i32> <i32 1, i32 0>
define amdgpu_kernel void @shufflevector_10_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %vaddr) {
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %vaddr
  %shuf = shufflevector <2 x i16> %vec, <2 x i16> undef, <2 x i32> <i32 1, i32 0>
  store <2 x i16> %shuf, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: 'shufflevector_11_v2i16'
; GFX9: estimated cost of 0 for {{.*}} shufflevector <2 x i16> %vec, <2 x i16> undef, <2 x i32> <i32 1, i32 1>
; VI: estimated cost of 2 for {{.*}} shufflevector <2 x i16> %vec, <2 x i16> undef, <2 x i32> <i32 1, i32 1>
define amdgpu_kernel void @shufflevector_11_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %vaddr) {
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %vaddr
  %shuf = shufflevector <2 x i16> %vec, <2 x i16> undef, <2 x i32> <i32 1, i32 1>
  store <2 x i16> %shuf, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: 'shufflevector_02_v2i16'
; GCN: estimated cost of 2 for {{.*}} shufflevector <2 x i16> %vec0, <2 x i16> %vec1, <2 x i32> <i32 0, i32 2>
define amdgpu_kernel void @shufflevector_02_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %vaddr0, <2 x i16> addrspace(1)* %vaddr1) {
  %vec0 = load <2 x i16>, <2 x i16> addrspace(1)* %vaddr0
  %vec1 = load <2 x i16>, <2 x i16> addrspace(1)* %vaddr1
  %shuf = shufflevector <2 x i16> %vec0, <2 x i16> %vec1, <2 x i32> <i32 0, i32 2>
  store <2 x i16> %shuf, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: 'shufflevector_xxx'
; TPT: Unknown cost for {{.*}} shufflevector <2 x i8> %vec, <2 x i8> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
; CS: estimated cost of 1 for {{.*}} shufflevector <2 x i8> %vec, <2 x i8> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
; Should not assert
define amdgpu_kernel void @shufflevector_xxx(<4 x i8> addrspace(1)* %out, <2 x i8> addrspace(1)* %vaddr) {
  %vec = load <2 x i8>, <2 x i8> addrspace(1)* %vaddr
  %shuf = shufflevector <2 x i8> %vec, <2 x i8> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  store <4 x i8> %shuf, <4 x i8> addrspace(1)* %out
  ret void
}
