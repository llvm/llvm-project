; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Make sure 64-bit BFE pattern does a 32-bit BFE on the relevant half.

; Extract the high bit of the low half
; GCN-LABEL: {{^}}v_uextract_bit_31_i64:
; GCN: buffer_load_dword [[VAL:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; GCN-DAG: v_lshrrev_b32_e32 v[[SHIFT:[0-9]+]], 31, [[VAL]]
; GCN-DAG: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[SHIFT]]:[[ZERO]]{{\]}}
define amdgpu_kernel void @v_uextract_bit_31_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i64, i64 addrspace(1)* %in.gep
  %srl = lshr i64 %ld.64, 31
  %bit = and i64 %srl, 1
  store i64 %bit, i64 addrspace(1)* %out.gep
  ret void
}

; Extract the high bit of the high half
; GCN-LABEL: {{^}}v_uextract_bit_63_i64:
; GCN: buffer_load_dword [[VAL:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4{{$}}
; GCN-DAG: v_lshrrev_b32_e32 v[[SHIFT:[0-9]+]], 31, [[VAL]]
; GCN-DAG: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[SHIFT]]:[[ZERO]]{{\]}}
define amdgpu_kernel void @v_uextract_bit_63_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i64, i64 addrspace(1)* %in.gep
  %srl = lshr i64 %ld.64, 63
  %bit = and i64 %srl, 1
  store i64 %bit, i64 addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_uextract_bit_1_i64:
; GCN: buffer_load_dword [[VAL:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; GCN-DAG: v_bfe_u32 v[[BFE:[0-9]+]], [[VAL]], 1, 1
; GCN-DAG: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[BFE]]:[[ZERO]]{{\]}}
define amdgpu_kernel void @v_uextract_bit_1_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i64, i64 addrspace(1)* %in.gep
  %srl = lshr i64 %ld.64, 1
  %bit = and i64 %srl, 1
  store i64 %bit, i64 addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_uextract_bit_20_i64:
; GCN: buffer_load_dword [[VAL:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; GCN-DAG: v_bfe_u32 v[[BFE:[0-9]+]], [[VAL]], 20, 1
; GCN-DAG: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[BFE]]:[[ZERO]]{{\]}}
define amdgpu_kernel void @v_uextract_bit_20_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i64, i64 addrspace(1)* %in.gep
  %srl = lshr i64 %ld.64, 20
  %bit = and i64 %srl, 1
  store i64 %bit, i64 addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_uextract_bit_32_i64:
; GCN: buffer_load_dword [[VAL:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4{{$}}
; GCN-DAG: v_and_b32_e32 v[[AND:[0-9]+]], 1, [[VAL]]
; GCN-DAG: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[AND]]:[[ZERO]]{{\]}}
define amdgpu_kernel void @v_uextract_bit_32_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i64, i64 addrspace(1)* %in.gep
  %srl = lshr i64 %ld.64, 32
  %bit = and i64 %srl, 1
  store i64 %bit, i64 addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_uextract_bit_33_i64:
; GCN: buffer_load_dword [[VAL:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4{{$}}
; GCN-DAG: v_bfe_u32 v[[BFE:[0-9]+]], [[VAL]], 1, 1{{$}}
; GCN-DAG: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[SHIFT]]:[[ZERO]]{{\]}}
define amdgpu_kernel void @v_uextract_bit_33_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i64, i64 addrspace(1)* %in.gep
  %srl = lshr i64 %ld.64, 33
  %bit = and i64 %srl, 1
  store i64 %bit, i64 addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_uextract_bit_20_21_i64:
; GCN: buffer_load_dword [[VAL:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; GCN-DAG: v_bfe_u32 v[[BFE:[0-9]+]], [[VAL]], 20, 2
; GCN-DAG: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[BFE]]:[[ZERO]]{{\]}}
define amdgpu_kernel void @v_uextract_bit_20_21_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i64, i64 addrspace(1)* %in.gep
  %srl = lshr i64 %ld.64, 20
  %bit = and i64 %srl, 3
  store i64 %bit, i64 addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_uextract_bit_1_30_i64:
; GCN: buffer_load_dword [[VAL:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; GCN-DAG: v_bfe_u32 v[[BFE:[0-9]+]], [[VAL]], 1, 30
; GCN-DAG: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[BFE]]:[[ZERO]]{{\]}}
define amdgpu_kernel void @v_uextract_bit_1_30_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i64, i64 addrspace(1)* %in.gep
  %srl = lshr i64 %ld.64, 1
  %bit = and i64 %srl, 1073741823
  store i64 %bit, i64 addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_uextract_bit_1_31_i64:
; GCN: buffer_load_dword [[VAL:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; GCN-DAG: v_lshrrev_b32_e32 v[[SHIFT:[0-9]+]], 1, [[VAL]]
; GCN-DAG: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[SHIFT]]:[[ZERO]]{{\]}}
define amdgpu_kernel void @v_uextract_bit_1_31_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i64, i64 addrspace(1)* %in.gep
  %srl = lshr i64 %ld.64, 1
  %bit = and i64 %srl, 2147483647
  store i64 %bit, i64 addrspace(1)* %out.gep
  ret void
}

; Spans the dword boundary, so requires full shift
; GCN-LABEL: {{^}}v_uextract_bit_31_32_i64:
; GCN: buffer_load_dwordx2 [[VAL:v\[[0-9]+:[0-9]+\]]]
; GCN: v_lshr_b64 v{{\[}}[[SHRLO:[0-9]+]]:[[SHRHI:[0-9]+]]{{\]}}, [[VAL]], 31
; GCN-DAG: v_and_b32_e32 v[[AND:[0-9]+]], 3, v[[SHRLO]]{{$}}
; GCN-DAG: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[AND]]:[[ZERO]]{{\]}}
define amdgpu_kernel void @v_uextract_bit_31_32_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i64, i64 addrspace(1)* %in.gep
  %srl = lshr i64 %ld.64, 31
  %bit = and i64 %srl, 3
  store i64 %bit, i64 addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_uextract_bit_32_33_i64:
; GCN: buffer_load_dword [[VAL:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4{{$}}
; GCN-DAG: v_bfe_u32 v[[BFE:[0-9]+]], [[VAL]], 1, 2
; GCN-DAG: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[BFE]]:[[ZERO]]{{\]}}
define amdgpu_kernel void @v_uextract_bit_32_33_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i64, i64 addrspace(1)* %in.gep
  %srl = lshr i64 %ld.64, 33
  %bit = and i64 %srl, 3
  store i64 %bit, i64 addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_uextract_bit_30_60_i64:
; GCN: buffer_load_dwordx2 [[VAL:v\[[0-9]+:[0-9]+\]]]
; GCN: v_lshr_b64 v{{\[}}[[SHRLO:[0-9]+]]:[[SHRHI:[0-9]+]]{{\]}}, [[VAL]], 30
; GCN-DAG: v_and_b32_e32 v[[AND:[0-9]+]], 0x3fffffff, v[[SHRLO]]{{$}}
; GCN-DAG: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[AND]]:[[ZERO]]{{\]}}
define amdgpu_kernel void @v_uextract_bit_30_60_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i64, i64 addrspace(1)* %in.gep
  %srl = lshr i64 %ld.64, 30
  %bit = and i64 %srl, 1073741823
  store i64 %bit, i64 addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_uextract_bit_33_63_i64:
; GCN: buffer_load_dword [[VAL:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4{{$}}
; GCN-DAG: v_bfe_u32 v[[BFE:[0-9]+]], [[VAL]], 1, 30
; GCN-DAG: v_mov_b32_e32 v[[BFE:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[SHIFT]]:[[ZERO]]{{\]}}
define amdgpu_kernel void @v_uextract_bit_33_63_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i64, i64 addrspace(1)* %in.gep
  %srl = lshr i64 %ld.64, 33
  %bit = and i64 %srl, 1073741823
  store i64 %bit, i64 addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_uextract_bit_31_63_i64:
; GCN: buffer_load_dwordx2 [[VAL:v\[[0-9]+:[0-9]+\]]]
; GCN: v_lshr_b64 v{{\[}}[[SHRLO:[0-9]+]]:[[SHRHI:[0-9]+]]{{\]}}, [[VAL]], 31
; GCN-NEXT: v_mov_b32_e32 v[[SHRHI]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[SHRLO]]:[[SHRHI]]{{\]}}
define amdgpu_kernel void @v_uextract_bit_31_63_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i64, i64 addrspace(1)* %in.gep
  %srl = lshr i64 %ld.64, 31
  %and = and i64 %srl, 4294967295
  store i64 %and, i64 addrspace(1)* %out
  ret void
}

; trunc applied before and mask
; GCN-LABEL: {{^}}v_uextract_bit_31_i64_trunc_i32:
; GCN: buffer_load_dword [[VAL:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; GCN-DAG: v_lshrrev_b32_e32 v[[SHIFT:[0-9]+]], 31, [[VAL]]
; GCN: buffer_store_dword v[[SHIFT]]
define amdgpu_kernel void @v_uextract_bit_31_i64_trunc_i32(i32 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i32, i32 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i64, i64 addrspace(1)* %in.gep
  %srl = lshr i64 %ld.64, 31
  %trunc = trunc i64 %srl to i32
  %bit = and i32 %trunc, 1
  store i32 %bit, i32 addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_uextract_bit_3_i64_trunc_i32:
; GCN: buffer_load_dword [[VAL:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; GCN: v_bfe_u32 [[BFE:v[0-9]+]], [[VAL]], 3, 1{{$}}
; GCN: buffer_store_dword [[BFE]]
define amdgpu_kernel void @v_uextract_bit_3_i64_trunc_i32(i32 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i32, i32 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i64, i64 addrspace(1)* %in.gep
  %srl = lshr i64 %ld.64, 3
  %trunc = trunc i64 %srl to i32
  %bit = and i32 %trunc, 1
  store i32 %bit, i32 addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_uextract_bit_33_i64_trunc_i32:
; GCN: buffer_load_dword [[VAL:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4{{$}}
; GCN: v_bfe_u32 [[BFE:v[0-9]+]], [[VAL]], 1, 1{{$}}
; GCN: buffer_store_dword [[BFE]]
define amdgpu_kernel void @v_uextract_bit_33_i64_trunc_i32(i32 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i32, i32 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i64, i64 addrspace(1)* %in.gep
  %srl = lshr i64 %ld.64, 33
  %trunc = trunc i64 %srl to i32
  %bit = and i32 %trunc, 1
  store i32 %bit, i32 addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_uextract_bit_31_32_i64_trunc_i32:
; GCN: buffer_load_dwordx2 [[VAL:v\[[0-9]+:[0-9]+\]]]
; GCN: v_lshr_b64 v{{\[}}[[SHRLO:[0-9]+]]:[[SHRHI:[0-9]+]]{{\]}}, [[VAL]], 31
; GCN-NEXT: v_and_b32_e32 v[[SHRLO]], 3, v[[SHRLO]]
; GCN-NOT: v[[SHRLO]]
; GCN: buffer_store_dword v[[SHRLO]]
define amdgpu_kernel void @v_uextract_bit_31_32_i64_trunc_i32(i32 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i32, i32 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i64, i64 addrspace(1)* %in.gep
  %srl = lshr i64 %ld.64, 31
  %trunc = trunc i64 %srl to i32
  %bit = and i32 %trunc, 3
  store i32 %bit, i32 addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}and_not_mask_i64:
; GCN: buffer_load_dwordx2 v{{\[}}[[VALLO:[0-9]+]]:[[VALHI:[0-9]+]]{{\]}}
; GCN: v_mov_b32_e32 v[[SHRHI]], 0{{$}}
; GCN: v_lshrrev_b32_e32 [[SHR:v[0-9]+]], 20, v[[VALLO]]
; GCN-DAG: v_and_b32_e32 v[[SHRLO]], 4, [[SHR]]
; GCN-NOT: v[[SHRLO]]
; GCN-NOT: v[[SHRHI]]
; GCN: buffer_store_dwordx2 v{{\[}}[[SHRLO]]:[[SHRHI]]{{\]}}
define amdgpu_kernel void @and_not_mask_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i64, i64 addrspace(1)* %in.gep
  %srl = lshr i64 %ld.64, 20
  %bit = and i64 %srl, 4
  store i64 %bit, i64 addrspace(1)* %out.gep
  ret void
}

; The instruction count is the same with/without hasOneUse, but
; keeping the 32-bit and has a smaller encoding size than the bfe.

; GCN-LABEL: {{^}}v_uextract_bit_27_29_multi_use_shift_i64:
; GCN: buffer_load_dwordx2 [[VAL:v\[[0-9]+:[0-9]+\]]]
; GCN-DAG: v_lshr_b64 v{{\[}}[[SHRLO:[0-9]+]]:[[SHRHI:[0-9]+]]{{\]}}, [[VAL]], 27
; GCN-DAG: v_and_b32_e32 v[[AND:[0-9]+]], 3, v[[SHRLO]]
; GCN-DAG: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[SHRLO]]:[[SHRHI]]{{\]}}
; GCN: buffer_store_dwordx2 v{{\[}}[[AND]]:[[ZERO]]{{\]}}
define amdgpu_kernel void @v_uextract_bit_27_29_multi_use_shift_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i64, i64 addrspace(1)* %in.gep
  %srl = lshr i64 %ld.64, 27
  %bit = and i64 %srl, 3
  store volatile i64 %srl, i64 addrspace(1)* %out
  store volatile i64 %bit, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_uextract_bit_34_37_multi_use_shift_i64:
; GCN: buffer_load_dword [[VAL:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4{{$}}
; GCN: v_mov_b32_e32 v[[ZERO_SHR:[0-9]+]], 0{{$}}
; GCN: v_mov_b32_e32 v[[ZERO_BFE:[0-9]+]], v[[ZERO_SHR]]
; GCN-DAG: v_lshrrev_b32_e32 v[[SHR:[0-9]+]], 2, [[VAL]]
; GCN-DAG: v_bfe_u32 v[[BFE:[0-9]+]], [[VAL]], 2, 3
; GCN-DAG: buffer_store_dwordx2 v{{\[}}[[SHR]]:[[ZERO_SHR]]{{\]}}
; GCN: buffer_store_dwordx2 v{{\[}}[[BFE]]:[[ZERO_BFE]]{{\]}}
define amdgpu_kernel void @v_uextract_bit_34_37_multi_use_shift_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %id.x
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %id.x
  %ld.64 = load i64, i64 addrspace(1)* %in.gep
  %srl = lshr i64 %ld.64, 34
  %bit = and i64 %srl, 7
  store volatile i64 %srl, i64 addrspace(1)* %out
  store volatile i64 %bit, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_uextract_bit_33_36_use_upper_half_shift_i64:
; GCN: buffer_load_dword [[VAL:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4{{$}}
; GCN-DAG: v_bfe_u32 v[[BFE:[0-9]+]], [[VAL]], 1, 3
; GCN-DAG: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[BFE]]:[[ZERO]]{{\]}}
; GCN: buffer_store_dword v[[ZERO]]
define amdgpu_kernel void @v_uextract_bit_33_36_use_upper_half_shift_i64(i64 addrspace(1)* %out0, i32 addrspace(1)* %out1, i64 addrspace(1)* %in) #1 {
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %id.x
  %out0.gep = getelementptr i64, i64 addrspace(1)* %out0, i32 %id.x
  %out1.gep = getelementptr i32, i32 addrspace(1)* %out1, i32 %id.x
  %ld.64 = load i64, i64 addrspace(1)* %in.gep
  %srl = lshr i64 %ld.64, 33
  %bit = and i64 %srl, 7
  store volatile i64 %bit, i64 addrspace(1)* %out0.gep

  %srl.srl32 = lshr i64 %srl, 32
  %srl.hi = trunc i64 %srl.srl32 to i32
  store volatile i32 %srl.hi, i32 addrspace(1)* %out1.gep
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

declare i32 @llvm.amdgcn.workgroup.id.x() #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
