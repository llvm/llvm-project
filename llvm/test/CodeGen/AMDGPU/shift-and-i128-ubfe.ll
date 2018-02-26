; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Extract the high bit of the 1st quarter
; GCN-LABEL: {{^}}v_uextract_bit_31_i128:
; GCN: buffer_load_dword [[VAL:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}

; GCN: v_mov_b32_e32 v[[ZERO0:[0-9]+]], 0{{$}}
; GCN: v_mov_b32_e32 v[[ZERO1:[0-9]+]], v[[ZERO0]]{{$}}
; GCN: v_mov_b32_e32 v[[ZERO2:[0-9]+]], v[[ZERO0]]{{$}}
; GCN: v_lshrrev_b32_e32 v[[SHIFT:[0-9]+]], 31, [[VAL]]

; GCN: buffer_store_dwordx4 v{{\[}}[[SHIFT]]:[[ZERO2]]{{\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; GCN: s_endpgm
define amdgpu_kernel void @v_uextract_bit_31_i128(i128 addrspace(1)* %out, i128 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %in.gep = getelementptr i128, i128 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i128, i128 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i128, i128 addrspace(1)* %in.gep
  %srl = lshr i128 %ld.64, 31
  %bit = and i128 %srl, 1
  store i128 %bit, i128 addrspace(1)* %out.gep
  ret void
}

; Extract the high bit of the 2nd quarter
; GCN-LABEL: {{^}}v_uextract_bit_63_i128:
; GCN-DAG: buffer_load_dword [[VAL:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4{{$}}

; GCN-DAG: v_mov_b32_e32 v[[ZERO0:[0-9]+]], 0{{$}}
; GCN: v_mov_b32_e32 v[[ZERO1:[0-9]+]], v[[ZERO0]]{{$}}
; GCN: v_mov_b32_e32 v[[ZERO2:[0-9]+]], v[[ZERO0]]{{$}}
; GCN: v_mov_b32_e32 v[[ZERO3:[0-9]+]], v[[ZERO0]]{{$}}
; GCN-DAG: v_lshrrev_b32_e32 v[[SHIFT:[0-9]+]], 31, [[VAL]]

; GCN-DAG: buffer_store_dwordx4 v{{\[}}[[SHIFT]]:[[ZERO3]]{{\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; GCN: s_endpgm
define amdgpu_kernel void @v_uextract_bit_63_i128(i128 addrspace(1)* %out, i128 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i128, i128 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i128, i128 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i128, i128 addrspace(1)* %in.gep
  %srl = lshr i128 %ld.64, 63
  %bit = and i128 %srl, 1
  store i128 %bit, i128 addrspace(1)* %out.gep
  ret void
}

; Extract the high bit of the 3rd quarter
; GCN-LABEL: {{^}}v_uextract_bit_95_i128:
; GCN: buffer_load_dword [[VAL:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}

; GCN-DAG: v_mov_b32_e32 v[[ZERO0:[0-9]+]], 0{{$}}
; GCN: v_mov_b32_e32 v[[ZERO1:[0-9]+]], v[[ZERO0]]{{$}}
; GCN: v_mov_b32_e32 v[[ZERO2:[0-9]+]], v[[ZERO0]]{{$}}
; GCN-DAG: v_lshrrev_b32_e32 v[[SHIFT:[0-9]+]], 31, [[VAL]]

; GCN-DAG: buffer_store_dwordx4 v{{\[}}[[SHIFT]]:[[ZERO2]]{{\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; GCN: s_endpgm
define amdgpu_kernel void @v_uextract_bit_95_i128(i128 addrspace(1)* %out, i128 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %in.gep = getelementptr i128, i128 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i128, i128 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i128, i128 addrspace(1)* %in.gep
  %srl = lshr i128 %ld.64, 95
  %bit = and i128 %srl, 1
  store i128 %bit, i128 addrspace(1)* %out.gep
  ret void
}

; Extract the high bit of the 4th quarter
; GCN-LABEL: {{^}}v_uextract_bit_127_i128:
; GCN-DAG: buffer_load_dword [[VAL:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:12{{$}}

; GCN-DAG: v_mov_b32_e32 v[[ZERO0:[0-9]+]], 0{{$}}
; GCN: v_mov_b32_e32 v[[ZERO1:[0-9]+]], v[[ZERO0]]{{$}}
; GCN: v_mov_b32_e32 v[[ZERO2:[0-9]+]], v[[ZERO0]]{{$}}
; GCN: v_mov_b32_e32 v[[ZERO3:[0-9]+]], v[[ZERO0]]{{$}}
; GCN-DAG: v_lshrrev_b32_e32 v[[SHIFT:[0-9]+]], 31, [[VAL]]

; GCN-DAG: buffer_store_dwordx4 v{{\[}}[[SHIFT]]:[[ZERO3]]{{\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; GCN: s_endpgm
define amdgpu_kernel void @v_uextract_bit_127_i128(i128 addrspace(1)* %out, i128 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i128, i128 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i128, i128 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i128, i128 addrspace(1)* %in.gep
  %srl = lshr i128 %ld.64, 127
  %bit = and i128 %srl, 1
  store i128 %bit, i128 addrspace(1)* %out.gep
  ret void
}

; Spans more than 2 dword boundaries
; GCN-LABEL: {{^}}v_uextract_bit_34_100_i128:
; GCN-DAG: buffer_load_dwordx4 v{{\[}}[[VAL0:[0-9]+]]:[[VAL3:[0-9]+]]{{\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}

; GCN-DAG: v_lshl_b64 v{{\[}}[[SHLLO:[0-9]+]]:[[SHLHI:[0-9]+]]{{\]}}, v{{\[[0-9]+:[0-9]+\]}}, 30
; GCN-DAG: v_lshrrev_b32_e32 v[[ELT1PART:[0-9]+]], 2, v{{[[0-9]+}}
; GCN-DAG: v_bfe_u32 v[[ELT2PART:[0-9]+]], v[[VAL3]], 2, 2{{$}}
; GCN-DAG: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0{{$}}
; GCN-DAG: v_or_b32_e32 v[[OR0:[0-9]+]], v[[ELT1PART]], v[[SHLLO]]
; GCN-DAG: v_mov_b32_e32 v[[ZERO1:[0-9]+]], v[[ZERO]]{{$}}

; GCN-DAG: buffer_store_dwordx4 v{{\[}}[[OR0]]:[[ZERO1]]{{\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; GCN: s_endpgm
define amdgpu_kernel void @v_uextract_bit_34_100_i128(i128 addrspace(1)* %out, i128 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i128, i128 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i128, i128 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i128, i128 addrspace(1)* %in.gep
  %srl = lshr i128 %ld.64, 34
  %bit = and i128 %srl, 73786976294838206463
  store i128 %bit, i128 addrspace(1)* %out.gep
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

declare i32 @llvm.amdgcn.workgroup.id.x() #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
