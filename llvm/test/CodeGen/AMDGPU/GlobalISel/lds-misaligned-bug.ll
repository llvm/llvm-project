; RUN: llc -global-isel -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,ALIGNED %s
; RUN: llc -global-isel -march=amdgcn -mcpu=gfx1011 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,ALIGNED %s
; RUN: llc -global-isel -march=amdgcn -mcpu=gfx1012 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,ALIGNED %s
; RUN: llc -global-isel -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs -mattr=+cumode < %s | FileCheck -check-prefixes=GCN,ALIGNED %s
; RUN: llc -global-isel -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs -mattr=+cumode,+unaligned-access-mode < %s | FileCheck -check-prefixes=GCN,UNALIGNED %s

; GCN-LABEL: test_local_misaligned_v2:
; GCN-DAG: ds_read2_b32
; GCN-DAG: ds_write2_b32
define amdgpu_kernel void @test_local_misaligned_v2(i32 addrspace(3)* %arg) {
bb:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, i32 addrspace(3)* %arg, i32 %lid
  %ptr = bitcast i32 addrspace(3)* %gep to <2 x i32> addrspace(3)*
  %load = load <2 x i32>, <2 x i32> addrspace(3)* %ptr, align 4
  %v1 = extractelement <2 x i32> %load, i32 0
  %v2 = extractelement <2 x i32> %load, i32 1
  %v3 = insertelement <2 x i32> undef, i32 %v2, i32 0
  %v4 = insertelement <2 x i32> %v3, i32 %v1, i32 1
  store <2 x i32> %v4, <2 x i32> addrspace(3)* %ptr, align 4
  ret void
}

; GCN-LABEL: test_local_misaligned_v4:
; ALIGNED-DAG: ds_read2_b32
; ALIGNED-DAG: ds_read2_b32
; ALIGNED-DAG: ds_write2_b32
; ALIGNED-DAG: ds_write2_b32
; UNALIGNED-DAG: ds_read2_b64
; UNALIGNED-DAG: ds_write2_b64
define amdgpu_kernel void @test_local_misaligned_v4(i32 addrspace(3)* %arg) {
bb:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, i32 addrspace(3)* %arg, i32 %lid
  %ptr = bitcast i32 addrspace(3)* %gep to <4 x i32> addrspace(3)*
  %load = load <4 x i32>, <4 x i32> addrspace(3)* %ptr, align 4
  %v1 = extractelement <4 x i32> %load, i32 0
  %v2 = extractelement <4 x i32> %load, i32 1
  %v3 = extractelement <4 x i32> %load, i32 2
  %v4 = extractelement <4 x i32> %load, i32 3
  %v5 = insertelement <4 x i32> undef, i32 %v4, i32 0
  %v6 = insertelement <4 x i32> %v5, i32 %v3, i32 1
  %v7 = insertelement <4 x i32> %v6, i32 %v2, i32 2
  %v8 = insertelement <4 x i32> %v7, i32 %v1, i32 3
  store <4 x i32> %v8, <4 x i32> addrspace(3)* %ptr, align 4
  ret void
}

; GCN-LABEL: test_local_misaligned_v3:
; ALIGNED-DAG: ds_read2_b32
; ALIGNED-DAG: ds_read_b32
; ALIGNED-DAG: ds_write2_b32
; ALIGNED-DAG: ds_write_b32
; UNALIGNED-DAG: ds_read_b96
; UNALIGNED-DAG: ds_write_b96
define amdgpu_kernel void @test_local_misaligned_v3(i32 addrspace(3)* %arg) {
bb:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, i32 addrspace(3)* %arg, i32 %lid
  %ptr = bitcast i32 addrspace(3)* %gep to <3 x i32> addrspace(3)*
  %load = load <3 x i32>, <3 x i32> addrspace(3)* %ptr, align 4
  %v1 = extractelement <3 x i32> %load, i32 0
  %v2 = extractelement <3 x i32> %load, i32 1
  %v3 = extractelement <3 x i32> %load, i32 2
  %v5 = insertelement <3 x i32> undef, i32 %v3, i32 0
  %v6 = insertelement <3 x i32> %v5, i32 %v1, i32 1
  %v7 = insertelement <3 x i32> %v6, i32 %v2, i32 2
  store <3 x i32> %v7, <3 x i32> addrspace(3)* %ptr, align 4
  ret void
}

; GCN-LABEL: test_local_aligned_v2:
; GCN-DAG: ds_read_b64
; GCN-DAG: ds_write_b64
define amdgpu_kernel void @test_local_aligned_v2(i32 addrspace(3)* %arg) {
bb:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, i32 addrspace(3)* %arg, i32 %lid
  %ptr = bitcast i32 addrspace(3)* %gep to <2 x i32> addrspace(3)*
  %load = load <2 x i32>, <2 x i32> addrspace(3)* %ptr, align 8
  %v1 = extractelement <2 x i32> %load, i32 0
  %v2 = extractelement <2 x i32> %load, i32 1
  %v3 = insertelement <2 x i32> undef, i32 %v2, i32 0
  %v4 = insertelement <2 x i32> %v3, i32 %v1, i32 1
  store <2 x i32> %v4, <2 x i32> addrspace(3)* %ptr, align 8
  ret void
}

; GCN-LABEL: test_local_aligned_v3:
; GCN-DAG: ds_read_b96
; GCN-DAG: ds_write_b96
define amdgpu_kernel void @test_local_aligned_v3(i32 addrspace(3)* %arg) {
bb:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, i32 addrspace(3)* %arg, i32 %lid
  %ptr = bitcast i32 addrspace(3)* %gep to <3 x i32> addrspace(3)*
  %load = load <3 x i32>, <3 x i32> addrspace(3)* %ptr, align 16
  %v1 = extractelement <3 x i32> %load, i32 0
  %v2 = extractelement <3 x i32> %load, i32 1
  %v3 = extractelement <3 x i32> %load, i32 2
  %v5 = insertelement <3 x i32> undef, i32 %v3, i32 0
  %v6 = insertelement <3 x i32> %v5, i32 %v1, i32 1
  %v7 = insertelement <3 x i32> %v6, i32 %v2, i32 2
  store <3 x i32> %v7, <3 x i32> addrspace(3)* %ptr, align 16
  ret void
}

; GCN-LABEL: test_local_v4_aligned8:
; ALIGNED-DAG: ds_read2_b64
; ALIGNED-DAG: ds_write2_b64
; UNALIGNED-DAG: ds_read2_b64
; UNALIGNED-DAG: ds_write2_b64
define amdgpu_kernel void @test_local_v4_aligned8(i32 addrspace(3)* %arg) {
bb:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, i32 addrspace(3)* %arg, i32 %lid
  %ptr = bitcast i32 addrspace(3)* %gep to <4 x i32> addrspace(3)*
  %load = load <4 x i32>, <4 x i32> addrspace(3)* %ptr, align 8
  %v1 = extractelement <4 x i32> %load, i32 0
  %v2 = extractelement <4 x i32> %load, i32 1
  %v3 = extractelement <4 x i32> %load, i32 2
  %v4 = extractelement <4 x i32> %load, i32 3
  %v5 = insertelement <4 x i32> undef, i32 %v4, i32 0
  %v6 = insertelement <4 x i32> %v5, i32 %v3, i32 1
  %v7 = insertelement <4 x i32> %v6, i32 %v2, i32 2
  %v8 = insertelement <4 x i32> %v7, i32 %v1, i32 3
  store <4 x i32> %v8, <4 x i32> addrspace(3)* %ptr, align 8
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
