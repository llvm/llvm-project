; RUN: llc -mtriple=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,FUNC %s
; RUN: llc -mtriple=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,FUNC %s
; RUN: llc -mtriple=r600 -mcpu=redwood < %s | FileCheck -check-prefix=FUNC %s

@local_memory.local_mem = internal unnamed_addr addrspace(3) global [128 x i32] undef, align 4

@lds = addrspace(3) global [512 x i32] undef, align 4

; On SI we need to make sure that the base offset is a register and
; not an immediate.

; FUNC-LABEL: {{^}}load_i32_local_const_ptr:
; GCN: v_mov_b32_e32 v[[PTR:[0-9]+]], 0{{$}}
; GCN: ds_read_b32 v{{[0-9]+}}, v[[PTR]] offset:4

; R600: LDS_READ_RET
define amdgpu_kernel void @load_i32_local_const_ptr(ptr addrspace(1) %out, ptr addrspace(3) %in) #0 {
entry:
  %tmp0 = getelementptr [512 x i32], ptr addrspace(3) @lds, i32 0, i32 1
  %tmp1 = load i32, ptr addrspace(3) %tmp0
  %tmp2 = getelementptr i32, ptr addrspace(1) %out, i32 1
  store i32 %tmp1, ptr addrspace(1) %tmp2
  ret void
}

; Test loading a i32 and v2i32 value from the same base pointer.
; FUNC-LABEL: {{^}}load_i32_v2i32_local:
; R600: LDS_READ_RET
; R600: LDS_READ_RET
; R600: LDS_READ_RET
; GCN-DAG: ds_read_b32
; GCN-DAG: ds_read2_b32
define amdgpu_kernel void @load_i32_v2i32_local(ptr addrspace(1) %out, ptr addrspace(3) %in) #0 {
  %scalar = load i32, ptr addrspace(3) %in
  %vec_ptr = getelementptr <2 x i32>, ptr addrspace(3) %in, i32 2
  %vec0 = load <2 x i32>, ptr addrspace(3) %vec_ptr, align 4
  %vec1 = insertelement <2 x i32> <i32 0, i32 0>, i32 %scalar, i32 0
  %vec = add <2 x i32> %vec0, %vec1
  store <2 x i32> %vec, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind }
