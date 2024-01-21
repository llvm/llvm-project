; RUN: llc -mtriple=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,ALIGNED,SPLIT %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1011 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,ALIGNED,SPLIT %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1012 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,ALIGNED,SPLIT %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1010 -verify-machineinstrs -mattr=+cumode < %s | FileCheck -check-prefixes=GCN,ALIGNED,VECT %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1010 -verify-machineinstrs -mattr=+cumode,+unaligned-access-mode < %s | FileCheck -check-prefixes=GCN,UNALIGNED,VECT %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,ALIGNED,VECT %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -verify-machineinstrs -mattr=+cumode < %s | FileCheck -check-prefixes=GCN,ALIGNED,VECT %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -verify-machineinstrs -mattr=+cumode,+unaligned-access-mode < %s | FileCheck -check-prefixes=GCN,UNALIGNED,VECT %s

; GCN-LABEL: test_local_misaligned_v2:
; GCN-DAG: ds_{{read2|load_2addr}}_b32
; GCN-DAG: ds_{{write2|store_2addr}}_b32
define amdgpu_kernel void @test_local_misaligned_v2(ptr addrspace(3) %arg) {
bb:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr addrspace(3) %arg, i32 %lid
  %load = load <2 x i32>, ptr addrspace(3) %gep, align 4
  %v1 = extractelement <2 x i32> %load, i32 0
  %v2 = extractelement <2 x i32> %load, i32 1
  %v3 = insertelement <2 x i32> undef, i32 %v2, i32 0
  %v4 = insertelement <2 x i32> %v3, i32 %v1, i32 1
  store <2 x i32> %v4, ptr addrspace(3) %gep, align 4
  ret void
}

; GCN-LABEL: test_local_misaligned_v4:
; GCN-DAG: ds_{{read2|load_2addr}}_b32
; GCN-DAG: ds_{{read2|load_2addr}}_b32
; GCN-DAG: ds_{{write2|store_2addr}}_b32
; GCN-DAG: ds_{{write2|store_2addr}}_b32
define amdgpu_kernel void @test_local_misaligned_v4(ptr addrspace(3) %arg) {
bb:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr addrspace(3) %arg, i32 %lid
  %load = load <4 x i32>, ptr addrspace(3) %gep, align 4
  %v1 = extractelement <4 x i32> %load, i32 0
  %v2 = extractelement <4 x i32> %load, i32 1
  %v3 = extractelement <4 x i32> %load, i32 2
  %v4 = extractelement <4 x i32> %load, i32 3
  %v5 = insertelement <4 x i32> undef, i32 %v4, i32 0
  %v6 = insertelement <4 x i32> %v5, i32 %v3, i32 1
  %v7 = insertelement <4 x i32> %v6, i32 %v2, i32 2
  %v8 = insertelement <4 x i32> %v7, i32 %v1, i32 3
  store <4 x i32> %v8, ptr addrspace(3) %gep, align 4
  ret void
}

; GCN-LABEL: test_local_misaligned_v3:
; GCN-DAG: ds_{{read2|load_2addr}}_b32
; GCN-DAG: ds_{{read|load}}_b32
; GCN-DAG: ds_{{write2|store_2addr}}_b32
; GCN-DAG: ds_{{write|store}}_b32
define amdgpu_kernel void @test_local_misaligned_v3(ptr addrspace(3) %arg) {
bb:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr addrspace(3) %arg, i32 %lid
  %load = load <3 x i32>, ptr addrspace(3) %gep, align 4
  %v1 = extractelement <3 x i32> %load, i32 0
  %v2 = extractelement <3 x i32> %load, i32 1
  %v3 = extractelement <3 x i32> %load, i32 2
  %v5 = insertelement <3 x i32> undef, i32 %v3, i32 0
  %v6 = insertelement <3 x i32> %v5, i32 %v1, i32 1
  %v7 = insertelement <3 x i32> %v6, i32 %v2, i32 2
  store <3 x i32> %v7, ptr addrspace(3) %gep, align 4
  ret void
}

; GCN-LABEL: test_flat_misaligned_v2:
; VECT-DAG:  flat_load_{{dwordx2|b64}} v
; VECT-DAG:  flat_store_{{dwordx2|b64}} v
; SPLIT-DAG: flat_load_{{dword|b32}} v
; SPLIT-DAG: flat_load_{{dword|b32}} v
; SPLIT-DAG: flat_store_{{dword|b32}} v
; SPLIT-DAG: flat_store_{{dword|b32}} v
define amdgpu_kernel void @test_flat_misaligned_v2(ptr %arg) {
bb:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr %arg, i32 %lid
  %load = load <2 x i32>, ptr %gep, align 4
  %v1 = extractelement <2 x i32> %load, i32 0
  %v2 = extractelement <2 x i32> %load, i32 1
  %v3 = insertelement <2 x i32> undef, i32 %v2, i32 0
  %v4 = insertelement <2 x i32> %v3, i32 %v1, i32 1
  store <2 x i32> %v4, ptr %gep, align 4
  ret void
}

; GCN-LABEL: test_flat_misaligned_v4:
; VECT-DAG:  flat_load_{{dwordx4|b128}} v
; VECT-DAG:  flat_store_{{dwordx4|b128}} v
; SPLIT-DAG: flat_load_{{dword|b32}} v
; SPLIT-DAG: flat_load_{{dword|b32}} v
; SPLIT-DAG: flat_load_{{dword|b32}} v
; SPLIT-DAG: flat_load_{{dword|b32}} v
; SPLIT-DAG: flat_store_{{dword|b32}} v
; SPLIT-DAG: flat_store_{{dword|b32}} v
; SPLIT-DAG: flat_store_{{dword|b32}} v
; SPLIT-DAG: flat_store_{{dword|b32}} v
define amdgpu_kernel void @test_flat_misaligned_v4(ptr %arg) {
bb:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr %arg, i32 %lid
  %load = load <4 x i32>, ptr %gep, align 4
  %v1 = extractelement <4 x i32> %load, i32 0
  %v2 = extractelement <4 x i32> %load, i32 1
  %v3 = extractelement <4 x i32> %load, i32 2
  %v4 = extractelement <4 x i32> %load, i32 3
  %v5 = insertelement <4 x i32> undef, i32 %v4, i32 0
  %v6 = insertelement <4 x i32> %v5, i32 %v3, i32 1
  %v7 = insertelement <4 x i32> %v6, i32 %v2, i32 2
  %v8 = insertelement <4 x i32> %v7, i32 %v1, i32 3
  store <4 x i32> %v8, ptr %gep, align 4
  ret void
}

; GCN-LABEL: test_flat_misaligned_v3:
; VECT-DAG:  flat_load_{{dwordx3|b96}} v
; VECT-DAG:  flat_store_{{dwordx3|b96}} v
; SPLIT-DAG: flat_load_{{dword|b32}} v
; SPLIT-DAG: flat_load_{{dword|b32}} v
; SPLIT-DAG: flat_load_{{dword|b32}} v
; SPLIT-DAG: flat_store_{{dword|b32}} v
; SPLIT-DAG: flat_store_{{dword|b32}} v
; SPLIT-DAG: flat_store_{{dword|b32}} v
define amdgpu_kernel void @test_flat_misaligned_v3(ptr %arg) {
bb:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr %arg, i32 %lid
  %load = load <3 x i32>, ptr %gep, align 4
  %v1 = extractelement <3 x i32> %load, i32 0
  %v2 = extractelement <3 x i32> %load, i32 1
  %v3 = extractelement <3 x i32> %load, i32 2
  %v5 = insertelement <3 x i32> undef, i32 %v3, i32 0
  %v6 = insertelement <3 x i32> %v5, i32 %v1, i32 1
  %v7 = insertelement <3 x i32> %v6, i32 %v2, i32 2
  store <3 x i32> %v7, ptr %gep, align 4
  ret void
}

; GCN-LABEL: test_local_aligned_v2:
; GCN-DAG: ds_{{read|load}}_b64
; GCN-DAG: ds_{{write|store}}_b64
define amdgpu_kernel void @test_local_aligned_v2(ptr addrspace(3) %arg) {
bb:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr addrspace(3) %arg, i32 %lid
  %load = load <2 x i32>, ptr addrspace(3) %gep, align 8
  %v1 = extractelement <2 x i32> %load, i32 0
  %v2 = extractelement <2 x i32> %load, i32 1
  %v3 = insertelement <2 x i32> undef, i32 %v2, i32 0
  %v4 = insertelement <2 x i32> %v3, i32 %v1, i32 1
  store <2 x i32> %v4, ptr addrspace(3) %gep, align 8
  ret void
}

; GCN-LABEL: test_local_aligned_v3:
; GCN-DAG: ds_{{read|load}}_b96
; GCN-DAG: ds_{{write|store}}_b96
define amdgpu_kernel void @test_local_aligned_v3(ptr addrspace(3) %arg) {
bb:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr addrspace(3) %arg, i32 %lid
  %load = load <3 x i32>, ptr addrspace(3) %gep, align 16
  %v1 = extractelement <3 x i32> %load, i32 0
  %v2 = extractelement <3 x i32> %load, i32 1
  %v3 = extractelement <3 x i32> %load, i32 2
  %v5 = insertelement <3 x i32> undef, i32 %v3, i32 0
  %v6 = insertelement <3 x i32> %v5, i32 %v1, i32 1
  %v7 = insertelement <3 x i32> %v6, i32 %v2, i32 2
  store <3 x i32> %v7, ptr addrspace(3) %gep, align 16
  ret void
}

; GCN-LABEL: test_flat_aligned_v2:
; GCN-DAG: flat_load_{{dwordx2|b64}} v
; GCN-DAG: flat_store_{{dwordx2|b64}} v
define amdgpu_kernel void @test_flat_aligned_v2(ptr %arg) {
bb:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr %arg, i32 %lid
  %load = load <2 x i32>, ptr %gep, align 8
  %v1 = extractelement <2 x i32> %load, i32 0
  %v2 = extractelement <2 x i32> %load, i32 1
  %v3 = insertelement <2 x i32> undef, i32 %v2, i32 0
  %v4 = insertelement <2 x i32> %v3, i32 %v1, i32 1
  store <2 x i32> %v4, ptr %gep, align 8
  ret void
}

; GCN-LABEL: test_flat_aligned_v4:
; GCN-DAG: flat_load_{{dwordx4|b128}} v
; GCN-DAG: flat_store_{{dwordx4|b128}} v
define amdgpu_kernel void @test_flat_aligned_v4(ptr %arg) {
bb:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr %arg, i32 %lid
  %load = load <4 x i32>, ptr %gep, align 16
  %v1 = extractelement <4 x i32> %load, i32 0
  %v2 = extractelement <4 x i32> %load, i32 1
  %v3 = extractelement <4 x i32> %load, i32 2
  %v4 = extractelement <4 x i32> %load, i32 3
  %v5 = insertelement <4 x i32> undef, i32 %v4, i32 0
  %v6 = insertelement <4 x i32> %v5, i32 %v3, i32 1
  %v7 = insertelement <4 x i32> %v6, i32 %v2, i32 2
  %v8 = insertelement <4 x i32> %v7, i32 %v1, i32 3
  store <4 x i32> %v8, ptr %gep, align 16
  ret void
}

; GCN-LABEL: test_local_v4_aligned8:
; ALIGNED-DAG: ds_{{read2|load_2addr}}_b64
; ALIGNED-DAG: ds_{{write2|store_2addr}}_b64
; UNALIGNED-DAG: ds_{{read2|load_2addr}}_b64
; UNALIGNED-DAG: ds_{{write2|store_2addr}}_b64
define amdgpu_kernel void @test_local_v4_aligned8(ptr addrspace(3) %arg) {
bb:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr addrspace(3) %arg, i32 %lid
  %load = load <4 x i32>, ptr addrspace(3) %gep, align 8
  %v1 = extractelement <4 x i32> %load, i32 0
  %v2 = extractelement <4 x i32> %load, i32 1
  %v3 = extractelement <4 x i32> %load, i32 2
  %v4 = extractelement <4 x i32> %load, i32 3
  %v5 = insertelement <4 x i32> undef, i32 %v4, i32 0
  %v6 = insertelement <4 x i32> %v5, i32 %v3, i32 1
  %v7 = insertelement <4 x i32> %v6, i32 %v2, i32 2
  %v8 = insertelement <4 x i32> %v7, i32 %v1, i32 3
  store <4 x i32> %v8, ptr addrspace(3) %gep, align 8
  ret void
}

; GCN-LABEL: test_flat_v4_aligned8:
; VECT-DAG:  flat_load_{{dwordx4|b128}} v
; VECT-DAG:  flat_store_{{dwordx4|b128}} v
; SPLIT-DAG: flat_load_{{dwordx2|b64}} v
; SPLIT-DAG: flat_load_{{dwordx2|b64}} v
; SPLIT-DAG: flat_store_{{dwordx2|b64}} v
; SPLIT-DAG: flat_store_{{dwordx2|b64}} v
define amdgpu_kernel void @test_flat_v4_aligned8(ptr %arg) {
bb:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, ptr %arg, i32 %lid
  %load = load <4 x i32>, ptr %gep, align 8
  %v1 = extractelement <4 x i32> %load, i32 0
  %v2 = extractelement <4 x i32> %load, i32 1
  %v3 = extractelement <4 x i32> %load, i32 2
  %v4 = extractelement <4 x i32> %load, i32 3
  %v5 = insertelement <4 x i32> undef, i32 %v4, i32 0
  %v6 = insertelement <4 x i32> %v5, i32 %v3, i32 1
  %v7 = insertelement <4 x i32> %v6, i32 %v2, i32 2
  %v8 = insertelement <4 x i32> %v7, i32 %v1, i32 3
  store <4 x i32> %v8, ptr %gep, align 8
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
