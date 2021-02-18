; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

; Check that register coalescer does not create an odd subreg when register tuples
; must be aligned.

; GCN-LABEL: {{^}}test_odd_int4:
; GCN:     global_load_dwordx4 v[{{[0-9]*[02468]:[0-9]*[13579]}}], v{{[0-9]+}}, s[{{[0-9:]+}}]
; GCN-DAG: v_mov_b32_e32 v{{[0-9]*}}[[LO:[02468]]], v{{[0-9]+}}
; GCN-DAG: v_mov_b32_e32 v{{[0-9]*}}[[HI:[13579]]], v{{[0-9]+}}
; GCN:     global_store_dwordx2 v{{[0-9]+}}, v{{\[}}[[LO]]:[[HI]]], s[{{[0-9:]+}}]

define amdgpu_kernel void @test_odd_int4(<4 x i32> addrspace(1)* %arg, <2 x i32> addrspace(1)* %arg1) {
bb:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep1 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %arg, i32 %lid
  %load = load <4 x i32>, <4 x i32> addrspace(1)* %gep1, align 16
  %shuffle = shufflevector <4 x i32> %load, <4 x i32> undef, <2 x i32> <i32 1, i32 3>
  %gep2 = getelementptr inbounds <2 x i32>, <2 x i32> addrspace(1)* %arg1, i32 %lid
  store <2 x i32> %shuffle, <2 x i32> addrspace(1)* %gep2, align 8
  ret void
}

; GCN-LABEL: {{^}}test_vector_creation:
; GCN:     global_load_dwordx2 v[{{[0-9]*[02468]}}:{{[0-9]+}}],
; GCN-DAG: v_mov_b32_e32 v{{[0-9]*}}[[LO:[02468]]], v{{[0-9]+}}
; GCN-DAG: v_mov_b32_e32 v{{[0-9]*}}[[HI:[13579]]], v{{[0-9]+}}
; GCN:     global_store_dwordx4 v[{{[0-9]*[02468]:[0-9]*[13579]}}], v[{{[0-9]*[02468]:[0-9]*[13579]}}]
define amdgpu_kernel void @test_vector_creation() {
entry:
  %tmp231 = load <4 x i16>, <4 x i16> addrspace(1)* undef, align 2
  %vext466 = shufflevector <4 x i16> %tmp231, <4 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  %vecinit467 = shufflevector <8 x i16> undef, <8 x i16> %vext466, <8 x i32> <i32 0, i32 1, i32 8, i32 9, i32 10, i32 11, i32 undef, i32 undef>
  %vecinit471 = shufflevector <8 x i16> %vecinit467, <8 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  store <8 x i16> %vecinit471, <8 x i16> addrspace(1)* undef, align 16
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
