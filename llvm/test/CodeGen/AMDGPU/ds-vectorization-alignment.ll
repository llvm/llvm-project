; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck --enable-var-scope --check-prefix=GCN %s

; Check that vectorizer does not create slow misaligned loads

; GCN-LABEL: {{^}}ds1align1:
; GCN-COUNT-2: ds_read_u8
; GCN-COUNT-2: ds_write_b8
define amdgpu_kernel void @ds1align1(ptr addrspace(3) %in, ptr addrspace(3) %out) {
  %val1 = load i8, ptr addrspace(3) %in, align 1
  %gep1 = getelementptr i8, ptr addrspace(3) %in, i32 1
  %val2 = load i8, ptr addrspace(3) %gep1, align 1
  store i8 %val1, ptr addrspace(3) %out, align 1
  %gep2 = getelementptr i8, ptr addrspace(3) %out, i32 1
  store i8 %val2, ptr addrspace(3) %gep2, align 1
  ret void
}

; GCN-LABEL: {{^}}ds2align2:
; GCN-COUNT-2: ds_read_u16
; GCN-COUNT-2: ds_write_b16
define amdgpu_kernel void @ds2align2(ptr addrspace(3) %in, ptr addrspace(3) %out) {
  %val1 = load i16, ptr addrspace(3) %in, align 2
  %gep1 = getelementptr i16, ptr addrspace(3) %in, i32 1
  %val2 = load i16, ptr addrspace(3) %gep1, align 2
  store i16 %val1, ptr addrspace(3) %out, align 2
  %gep2 = getelementptr i16, ptr addrspace(3) %out, i32 1
  store i16 %val2, ptr addrspace(3) %gep2, align 2
  ret void
}

; GCN-LABEL: {{^}}ds4align4:
; GCN: ds_read2_b32
; GCN: ds_write2_b32
define amdgpu_kernel void @ds4align4(ptr addrspace(3) %in, ptr addrspace(3) %out) {
  %val1 = load i32, ptr addrspace(3) %in, align 4
  %gep1 = getelementptr i32, ptr addrspace(3) %in, i32 1
  %val2 = load i32, ptr addrspace(3) %gep1, align 4
  store i32 %val1, ptr addrspace(3) %out, align 4
  %gep2 = getelementptr i32, ptr addrspace(3) %out, i32 1
  store i32 %val2, ptr addrspace(3) %gep2, align 4
  ret void
}

; GCN-LABEL: {{^}}ds8align8:
; GCN: ds_read2_b64
; GCN: ds_write2_b64
define amdgpu_kernel void @ds8align8(ptr addrspace(3) %in, ptr addrspace(3) %out) {
  %val1 = load i64, ptr addrspace(3) %in, align 8
  %gep1 = getelementptr i64, ptr addrspace(3) %in, i64 1
  %val2 = load i64, ptr addrspace(3) %gep1, align 8
  store i64 %val1, ptr addrspace(3) %out, align 8
  %gep2 = getelementptr i64, ptr addrspace(3) %out, i64 1
  store i64 %val2, ptr addrspace(3) %gep2, align 8
  ret void
}

; GCN-LABEL: {{^}}ds1align2:
; GCN: ds_read_u16
; GCN: ds_write_b16
define amdgpu_kernel void @ds1align2(ptr addrspace(3) %in, ptr addrspace(3) %out) {
  %val1 = load i8, ptr addrspace(3) %in, align 2
  %gep1 = getelementptr i8, ptr addrspace(3) %in, i32 1
  %val2 = load i8, ptr addrspace(3) %gep1, align 2
  store i8 %val1, ptr addrspace(3) %out, align 2
  %gep2 = getelementptr i8, ptr addrspace(3) %out, i32 1
  store i8 %val2, ptr addrspace(3) %gep2, align 2
  ret void
}

; GCN-LABEL: {{^}}ds2align4:
; GCN: ds_read_b32
; GCN: ds_write_b32
define amdgpu_kernel void @ds2align4(ptr addrspace(3) %in, ptr addrspace(3) %out) {
  %val1 = load i16, ptr addrspace(3) %in, align 4
  %gep1 = getelementptr i16, ptr addrspace(3) %in, i32 1
  %val2 = load i16, ptr addrspace(3) %gep1, align 4
  store i16 %val1, ptr addrspace(3) %out, align 4
  %gep2 = getelementptr i16, ptr addrspace(3) %out, i32 1
  store i16 %val2, ptr addrspace(3) %gep2, align 4
  ret void
}

; GCN-LABEL: {{^}}ds4align8:
; GCN: ds_read_b64
; GCN: ds_write_b64
define amdgpu_kernel void @ds4align8(ptr addrspace(3) %in, ptr addrspace(3) %out) {
  %val1 = load i32, ptr addrspace(3) %in, align 8
  %gep1 = getelementptr i32, ptr addrspace(3) %in, i32 1
  %val2 = load i32, ptr addrspace(3) %gep1, align 8
  store i32 %val1, ptr addrspace(3) %out, align 8
  %gep2 = getelementptr i32, ptr addrspace(3) %out, i32 1
  store i32 %val2, ptr addrspace(3) %gep2, align 8
  ret void
}

; GCN-LABEL: {{^}}ds8align16:
; GCN: ds_read_b128
; GCN: ds_write_b128
define amdgpu_kernel void @ds8align16(ptr addrspace(3) %in, ptr addrspace(3) %out) {
  %val1 = load i64, ptr addrspace(3) %in, align 16
  %gep1 = getelementptr i64, ptr addrspace(3) %in, i64 1
  %val2 = load i64, ptr addrspace(3) %gep1, align 16
  store i64 %val1, ptr addrspace(3) %out, align 16
  %gep2 = getelementptr i64, ptr addrspace(3) %out, i64 1
  store i64 %val2, ptr addrspace(3) %gep2, align 16
  ret void
}
