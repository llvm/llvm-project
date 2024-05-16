; RUN: llc -mtriple=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SI %s
; RUN: llc -mtriple=amdgcn -mcpu=hawaii -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,NOSI %s

@compute_lds = external addrspace(3) global [512 x i32], align 16

; GCN-LABEL: {{^}}store_aligned:
; GCN: ds_write_b64
define amdgpu_cs void @store_aligned(ptr addrspace(3) %ptr) #0 {
entry:
  %ptr.gep.1 = getelementptr i32, ptr addrspace(3) %ptr, i32 1

  store i32 42, ptr addrspace(3) %ptr, align 8
  store i32 43, ptr addrspace(3) %ptr.gep.1
  ret void
}


; GCN-LABEL: {{^}}load_aligned:
; GCN: ds_read_b64
define amdgpu_cs <2 x float> @load_aligned(ptr addrspace(3) %ptr) #0 {
entry:
  %ptr.gep.1 = getelementptr i32, ptr addrspace(3) %ptr, i32 1

  %v.0 = load i32, ptr addrspace(3) %ptr, align 8
  %v.1 = load i32, ptr addrspace(3) %ptr.gep.1

  %r.0 = insertelement <2 x i32> undef, i32 %v.0, i32 0
  %r.1 = insertelement <2 x i32> %r.0, i32 %v.1, i32 1
  %bc = bitcast <2 x i32> %r.1 to <2 x float>
  ret <2 x float> %bc
}


; GCN-LABEL: {{^}}store_global_const_idx:
; GCN: ds_write2_b32
define amdgpu_cs void @store_global_const_idx() #0 {
entry:
  %ptr.a = getelementptr [512 x i32], ptr addrspace(3) @compute_lds, i32 0, i32 3
  %ptr.b = getelementptr [512 x i32], ptr addrspace(3) @compute_lds, i32 0, i32 4

  store i32 42, ptr addrspace(3) %ptr.a
  store i32 43, ptr addrspace(3) %ptr.b
  ret void
}


; GCN-LABEL: {{^}}load_global_const_idx:
; GCN: ds_read2_b32
define amdgpu_cs <2 x float> @load_global_const_idx() #0 {
entry:
  %ptr.a = getelementptr [512 x i32], ptr addrspace(3) @compute_lds, i32 0, i32 3
  %ptr.b = getelementptr [512 x i32], ptr addrspace(3) @compute_lds, i32 0, i32 4

  %v.0 = load i32, ptr addrspace(3) %ptr.a
  %v.1 = load i32, ptr addrspace(3) %ptr.b

  %r.0 = insertelement <2 x i32> undef, i32 %v.0, i32 0
  %r.1 = insertelement <2 x i32> %r.0, i32 %v.1, i32 1
  %bc = bitcast <2 x i32> %r.1 to <2 x float>
  ret <2 x float> %bc
}


; GCN-LABEL: {{^}}store_global_var_idx_case1:
; SI: ds_write_b32
; SI: ds_write_b32
; NOSI: ds_write2_b32
define amdgpu_cs void @store_global_var_idx_case1(i32 %idx) #0 {
entry:
  %ptr.a = getelementptr [512 x i32], ptr addrspace(3) @compute_lds, i32 0, i32 %idx
  %ptr.b = getelementptr i32, ptr addrspace(3) %ptr.a, i32 1

  store i32 42, ptr addrspace(3) %ptr.a
  store i32 43, ptr addrspace(3) %ptr.b
  ret void
}


; GCN-LABEL: {{^}}load_global_var_idx_case1:
; SI: ds_read_b32
; SI: ds_read_b32
; NOSI: ds_read2_b32
define amdgpu_cs <2 x float> @load_global_var_idx_case1(i32 %idx) #0 {
entry:
  %ptr.a = getelementptr [512 x i32], ptr addrspace(3) @compute_lds, i32 0, i32 %idx
  %ptr.b = getelementptr i32, ptr addrspace(3) %ptr.a, i32 1

  %v.0 = load i32, ptr addrspace(3) %ptr.a
  %v.1 = load i32, ptr addrspace(3) %ptr.b

  %r.0 = insertelement <2 x i32> undef, i32 %v.0, i32 0
  %r.1 = insertelement <2 x i32> %r.0, i32 %v.1, i32 1
  %bc = bitcast <2 x i32> %r.1 to <2 x float>
  ret <2 x float> %bc
}


; GCN-LABEL: {{^}}store_global_var_idx_case2:
; GCN: ds_write2_b32
define amdgpu_cs void @store_global_var_idx_case2(i32 %idx) #0 {
entry:
  %idx.and = and i32 %idx, 255
  %ptr.a = getelementptr [512 x i32], ptr addrspace(3) @compute_lds, i32 0, i32 %idx.and
  %ptr.b = getelementptr i32, ptr addrspace(3) %ptr.a, i32 1

  store i32 42, ptr addrspace(3) %ptr.a
  store i32 43, ptr addrspace(3) %ptr.b
  ret void
}


; GCN-LABEL: {{^}}load_global_var_idx_case2:
; GCN: ds_read2_b32
define amdgpu_cs <2 x float> @load_global_var_idx_case2(i32 %idx) #0 {
entry:
  %idx.and = and i32 %idx, 255
  %ptr.a = getelementptr [512 x i32], ptr addrspace(3) @compute_lds, i32 0, i32 %idx.and
  %ptr.b = getelementptr i32, ptr addrspace(3) %ptr.a, i32 1

  %v.0 = load i32, ptr addrspace(3) %ptr.a
  %v.1 = load i32, ptr addrspace(3) %ptr.b

  %r.0 = insertelement <2 x i32> undef, i32 %v.0, i32 0
  %r.1 = insertelement <2 x i32> %r.0, i32 %v.1, i32 1
  %bc = bitcast <2 x i32> %r.1 to <2 x float>
  ret <2 x float> %bc
}

attributes #0 = { nounwind }
