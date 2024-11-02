; RUN: llc -mtriple=amdgcn -verify-machineinstrs< %s | FileCheck -check-prefixes=SI,MUBUF,ALIGNED %s
; RUN: llc -mtriple=amdgcn -mcpu=bonaire -mattr=+unaligned-access-mode -verify-machineinstrs< %s | FileCheck -check-prefixes=SI,MUBUF,UNALIGNED %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs< %s | FileCheck -check-prefixes=SI,MUBUF,ALIGNED %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -mattr=+enable-flat-scratch -verify-machineinstrs < %s | FileCheck -check-prefixes=SI,FLATSCR,ALIGNED %s

; SI-LABEL: {{^}}local_unaligned_load_store_i16:
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_write_b8
; SI: ds_write_b8
; SI: s_endpgm
define amdgpu_kernel void @local_unaligned_load_store_i16(ptr addrspace(3) %p, ptr addrspace(3) %r) #0 {
  %v = load i16, ptr addrspace(3) %p, align 1
  store i16 %v, ptr addrspace(3) %r, align 1
  ret void
}

; SI-LABEL: {{^}}global_unaligned_load_store_i16:
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte

; UNALIGNED: buffer_load_ushort
; UNALIGNED: buffer_store_short
; SI: s_endpgm
define amdgpu_kernel void @global_unaligned_load_store_i16(ptr addrspace(1) %p, ptr addrspace(1) %r) #0 {
  %v = load i16, ptr addrspace(1) %p, align 1
  store i16 %v, ptr addrspace(1) %r, align 1
  ret void
}

; SI-LABEL: {{^}}local_unaligned_load_store_i32:

; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI-NOT: v_or
; SI-NOT: v_lshl
; SI: ds_write_b8
; SI: ds_write_b8
; SI: ds_write_b8
; SI: ds_write_b8
; SI: s_endpgm
define amdgpu_kernel void @local_unaligned_load_store_i32(ptr addrspace(3) %p, ptr addrspace(3) %r) #0 {
  %v = load i32, ptr addrspace(3) %p, align 1
  store i32 %v, ptr addrspace(3) %r, align 1
  ret void
}

; SI-LABEL: {{^}}global_unaligned_load_store_i32:
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte

; UNALIGNED: buffer_load_dword
; UNALIGNED: buffer_store_dword
define amdgpu_kernel void @global_unaligned_load_store_i32(ptr addrspace(1) %p, ptr addrspace(1) %r) #0 {
  %v = load i32, ptr addrspace(1) %p, align 1
  store i32 %v, ptr addrspace(1) %r, align 1
  ret void
}

; SI-LABEL: {{^}}global_align2_load_store_i32:
; ALIGNED: buffer_load_ushort
; ALIGNED: buffer_load_ushort
; ALIGNED: buffer_store_short
; ALIGNED: buffer_store_short

; UNALIGNED: buffer_load_dword
; UNALIGNED: buffer_store_dword
define amdgpu_kernel void @global_align2_load_store_i32(ptr addrspace(1) %p, ptr addrspace(1) %r) #0 {
  %v = load i32, ptr addrspace(1) %p, align 2
  store i32 %v, ptr addrspace(1) %r, align 2
  ret void
}

; GCN-LABEL: {{^}}local_align2_load_store_i32:
; GCN: ds_read_u16
; GCN: ds_read_u16
; GCN: ds_write_b16
; GCN: ds_write_b16
define amdgpu_kernel void @local_align2_load_store_i32(ptr addrspace(3) %p, ptr addrspace(3) %r) #0 {
  %v = load i32, ptr addrspace(3) %p, align 2
  store i32 %v, ptr addrspace(3) %r, align 2
  ret void
}

; SI-LABEL: {{^}}local_unaligned_load_store_i64:
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8

; SI-NOT: v_or_b32
; SI-NOT: v_lshl
; SI: ds_write_b8
; SI-NOT: v_or_b32
; SI-NOT: v_lshl

; SI: ds_write_b8
; SI-NOT: v_or_b32
; SI-NOT: v_lshl

; SI: ds_write_b8
; SI-NOT: v_or_b32
; SI-NOT: v_lshl

; SI: ds_write_b8
; SI-NOT: v_or_b32
; SI-NOT: v_lshl

; SI: ds_write_b8
; SI-NOT: v_or_b32
; SI-NOT: v_lshl

; SI: ds_write_b8
; SI-NOT: v_or_b32
; SI-NOT: v_lshl

; SI: ds_write_b8
; SI-NOT: v_or_b32
; SI-NOT: v_lshl
; SI: ds_write_b8
; SI: s_endpgm
define amdgpu_kernel void @local_unaligned_load_store_i64(ptr addrspace(3) %p, ptr addrspace(3) %r) #0 {
  %v = load i64, ptr addrspace(3) %p, align 1
  store i64 %v, ptr addrspace(3) %r, align 1
  ret void
}

; SI-LABEL: {{^}}local_unaligned_load_store_v2i32:
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8

; SI-NOT: v_or_b32
; SI-NOT: v_lshl
; SI: ds_write_b8
; SI-NOT: v_or_b32
; SI-NOT: v_lshl

; SI: ds_write_b8
; SI-NOT: v_or_b32
; SI-NOT: v_lshl

; SI: ds_write_b8
; SI-NOT: v_or_b32
; SI-NOT: v_lshl

; SI: ds_write_b8
; SI-NOT: v_or_b32
; SI-NOT: v_lshl

; SI: ds_write_b8
; SI-NOT: v_or_b32
; SI-NOT: v_lshl

; SI: ds_write_b8
; SI-NOT: v_or_b32
; SI-NOT: v_lshl

; SI: ds_write_b8
; SI-NOT: v_or_b32
; SI-NOT: v_lshl
; SI: ds_write_b8
; SI: s_endpgm
define amdgpu_kernel void @local_unaligned_load_store_v2i32(ptr addrspace(3) %p, ptr addrspace(3) %r) #0 {
  %v = load <2 x i32>, ptr addrspace(3) %p, align 1
  store <2 x i32> %v, ptr addrspace(3) %r, align 1
  ret void
}

; SI-LABEL: {{^}}global_align2_load_store_i64:
; ALIGNED: buffer_load_ushort
; ALIGNED: buffer_load_ushort

; ALIGNED-NOT: v_or_
; ALIGNED-NOT: v_lshl

; ALIGNED: buffer_load_ushort

; ALIGNED-NOT: v_or_
; ALIGNED-NOT: v_lshl

; ALIGNED: buffer_load_ushort

; ALIGNED-NOT: v_or_
; ALIGNED-NOT: v_lshl

; ALIGNED: buffer_store_short
; ALIGNED: buffer_store_short
; ALIGNED: buffer_store_short
; ALIGNED: buffer_store_short

; UNALIGNED: buffer_load_dwordx2
; UNALIGNED: buffer_store_dwordx2
define amdgpu_kernel void @global_align2_load_store_i64(ptr addrspace(1) %p, ptr addrspace(1) %r) #0 {
  %v = load i64, ptr addrspace(1) %p, align 2
  store i64 %v, ptr addrspace(1) %r, align 2
  ret void
}

; SI-LABEL: {{^}}unaligned_load_store_i64_global:
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte

; ALIGNED-NOT: v_or_
; ALIGNED-NOT: v_lshl

; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte

; UNALIGNED: buffer_load_dwordx2
; UNALIGNED: buffer_store_dwordx2
define amdgpu_kernel void @unaligned_load_store_i64_global(ptr addrspace(1) %p, ptr addrspace(1) %r) #0 {
  %v = load i64, ptr addrspace(1) %p, align 1
  store i64 %v, ptr addrspace(1) %r, align 1
  ret void
}

; GCN-LABEL: {{^}}local_unaligned_load_store_v4i32:
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8

; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8

; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8

; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8

; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: ds_write_b8

; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: ds_write_b8

; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: ds_write_b8

; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: s_endpgm
define amdgpu_kernel void @local_unaligned_load_store_v4i32(ptr addrspace(3) %p, ptr addrspace(3) %r) #0 {
  %v = load <4 x i32>, ptr addrspace(3) %p, align 1
  store <4 x i32> %v, ptr addrspace(3) %r, align 1
  ret void
}

; SI-LABEL: {{^}}global_unaligned_load_store_v4i32
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte

; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte
; ALIGNED: buffer_store_byte

; UNALIGNED: buffer_load_dwordx4
; UNALIGNED: buffer_store_dwordx4
define amdgpu_kernel void @global_unaligned_load_store_v4i32(ptr addrspace(1) %p, ptr addrspace(1) %r) #0 {
  %v = load <4 x i32>, ptr addrspace(1) %p, align 1
  store <4 x i32> %v, ptr addrspace(1) %r, align 1
  ret void
}

; GCN-LABEL: {{^}}local_load_i64_align_4:
; GCN: ds_read2_b32
define amdgpu_kernel void @local_load_i64_align_4(ptr addrspace(1) nocapture %out, ptr addrspace(3) %in) #0 {
  %val = load i64, ptr addrspace(3) %in, align 4
  store i64 %val, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}local_load_i64_align_4_with_offset
; GCN: ds_read2_b32 v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]}} offset0:8 offset1:9
define amdgpu_kernel void @local_load_i64_align_4_with_offset(ptr addrspace(1) nocapture %out, ptr addrspace(3) %in) #0 {
  %ptr = getelementptr i64, ptr addrspace(3) %in, i32 4
  %val = load i64, ptr addrspace(3) %ptr, align 4
  store i64 %val, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}local_load_i64_align_4_with_split_offset:
; The tests for the case where the lo offset is 8-bits, but the hi offset is 9-bits
; GCN: ds_read2_b32 v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]}} offset1:1
; GCN: s_endpgm
define amdgpu_kernel void @local_load_i64_align_4_with_split_offset(ptr addrspace(1) nocapture %out, ptr addrspace(3) %in) #0 {
  %ptr255 = getelementptr i32, ptr addrspace(3) %in, i32 255
  %val = load i64, ptr addrspace(3) %ptr255, align 4
  store i64 %val, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}local_load_i64_align_1:
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: store_dwordx2
define amdgpu_kernel void @local_load_i64_align_1(ptr addrspace(1) nocapture %out, ptr addrspace(3) %in) #0 {
  %val = load i64, ptr addrspace(3) %in, align 1
  store i64 %val, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}local_store_i64_align_4:
; GCN: ds_write2_b32
define amdgpu_kernel void @local_store_i64_align_4(ptr addrspace(3) %out, i64 %val) #0 {
  store i64 %val, ptr addrspace(3) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}local_store_i64_align_4_with_offset
; GCN: ds_write2_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset0:8 offset1:9
; GCN: s_endpgm
define amdgpu_kernel void @local_store_i64_align_4_with_offset(ptr addrspace(3) %out) #0 {
  %ptr = getelementptr i64, ptr addrspace(3) %out, i32 4
  store i64 0, ptr addrspace(3) %ptr, align 4
  ret void
}

; GCN-LABEL: {{^}}local_store_i64_align_4_with_split_offset:
; The tests for the case where the lo offset is 8-bits, but the hi offset is 9-bits
; GCN: ds_write2_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset1:1
; GCN: s_endpgm
define amdgpu_kernel void @local_store_i64_align_4_with_split_offset(ptr addrspace(3) %out) #0 {
  %ptr255 = getelementptr i32, ptr addrspace(3) %out, i32 255
  store i64 0, ptr addrspace(3) %out, align 4
  ret void
}

; SI-LABEL: {{^}}constant_unaligned_load_i32:
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte

; UNALIGNED: s_load_dword

; SI: buffer_store_dword
define amdgpu_kernel void @constant_unaligned_load_i32(ptr addrspace(4) %p, ptr addrspace(1) %r) #0 {
  %v = load i32, ptr addrspace(4) %p, align 1
  store i32 %v, ptr addrspace(1) %r, align 4
  ret void
}

; SI-LABEL: {{^}}constant_align2_load_i32:
; ALIGNED: buffer_load_ushort
; ALIGNED: buffer_load_ushort

; UNALIGNED: s_load_dword
; UNALIGNED: buffer_store_dword
define amdgpu_kernel void @constant_align2_load_i32(ptr addrspace(4) %p, ptr addrspace(1) %r) #0 {
  %v = load i32, ptr addrspace(4) %p, align 2
  store i32 %v, ptr addrspace(1) %r, align 4
  ret void
}

; SI-LABEL: {{^}}constant_align2_load_i64:
; ALIGNED: buffer_load_ushort
; ALIGNED: buffer_load_ushort
; ALIGNED: buffer_load_ushort
; ALIGNED: buffer_load_ushort

; UNALIGNED: s_load_dwordx4
; UNALIGNED: buffer_store_dwordx2
define amdgpu_kernel void @constant_align2_load_i64(ptr addrspace(4) %p, ptr addrspace(1) %r) #0 {
  %v = load i64, ptr addrspace(4) %p, align 2
  store i64 %v, ptr addrspace(1) %r, align 4
  ret void
}

; SI-LABEL: {{^}}constant_align4_load_i64:
; SI: s_load_dwordx2
; SI: buffer_store_dwordx2
define amdgpu_kernel void @constant_align4_load_i64(ptr addrspace(4) %p, ptr addrspace(1) %r) #0 {
  %v = load i64, ptr addrspace(4) %p, align 4
  store i64 %v, ptr addrspace(1) %r, align 4
  ret void
}

; SI-LABEL: {{^}}constant_align4_load_v4i32:
; SI: s_load_dwordx4
; SI: buffer_store_dwordx4
define amdgpu_kernel void @constant_align4_load_v4i32(ptr addrspace(4) %p, ptr addrspace(1) %r) #0 {
  %v = load <4 x i32>, ptr addrspace(4) %p, align 4
  store <4 x i32> %v, ptr addrspace(1) %r, align 4
  ret void
}

; SI-LABEL: {{^}}constant_unaligned_load_v2i32:
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte

; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte

; UNALIGNED: buffer_load_dwordx2

; SI: buffer_store_dwordx2
define amdgpu_kernel void @constant_unaligned_load_v2i32(ptr addrspace(4) %p, ptr addrspace(1) %r) #0 {
  %v = load <2 x i32>, ptr addrspace(4) %p, align 1
  store <2 x i32> %v, ptr addrspace(1) %r, align 4
  ret void
}

; SI-LABEL: {{^}}constant_unaligned_load_v4i32:
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte

; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte

; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte

; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte
; ALIGNED: buffer_load_ubyte

; UNALIGNED: buffer_load_dwordx4

; SI: buffer_store_dwordx4
define amdgpu_kernel void @constant_unaligned_load_v4i32(ptr addrspace(4) %p, ptr addrspace(1) %r) #0 {
  %v = load <4 x i32>, ptr addrspace(4) %p, align 1
  store <4 x i32> %v, ptr addrspace(1) %r, align 4
  ret void
}

; SI-LABEL: {{^}}constant_align4_load_i8:
; SI: s_load_dword
; SI: buffer_store_byte
define amdgpu_kernel void @constant_align4_load_i8(ptr addrspace(4) %p, ptr addrspace(1) %r) #0 {
  %v = load i8, ptr addrspace(4) %p, align 4
  store i8 %v, ptr addrspace(1) %r, align 4
  ret void
}

; SI-LABEL: {{^}}constant_align2_load_i8:
; SI: buffer_load_ubyte
; SI: buffer_store_byte
define amdgpu_kernel void @constant_align2_load_i8(ptr addrspace(4) %p, ptr addrspace(1) %r) #0 {
  %v = load i8, ptr addrspace(4) %p, align 2
  store i8 %v, ptr addrspace(1) %r, align 2
  ret void
}

; SI-LABEL: {{^}}constant_align4_merge_load_2_i32:
; SI: s_load_dwordx2 s[[[LO:[0-9]+]]:[[HI:[0-9]+]]], s{{\[[0-9]+:[0-9]+\]}}, 0x0{{$}}
; SI-DAG: v_mov_b32_e32 v[[VLO:[0-9]+]], s[[LO]]
; SI-DAG: v_mov_b32_e32 v[[VHI:[0-9]+]], s[[HI]]
; SI: buffer_store_dwordx2 v[[[VLO]]:[[VHI]]]
define amdgpu_kernel void @constant_align4_merge_load_2_i32(ptr addrspace(4) %p, ptr addrspace(1) %r) #0 {
  %gep0 = getelementptr i32, ptr addrspace(4) %p, i64 1
  %v0 = load i32, ptr addrspace(4) %p, align 4
  %v1 = load i32, ptr addrspace(4) %gep0, align 4

  %gep1 = getelementptr i32, ptr addrspace(1) %r, i64 1
  store i32 %v0, ptr addrspace(1) %r, align 4
  store i32 %v1, ptr addrspace(1) %gep1, align 4
  ret void
}

; SI-LABEL: {{^}}local_load_align1_v16i8:
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8

; SI: ScratchSize: 0{{$}}
define amdgpu_kernel void @local_load_align1_v16i8(ptr addrspace(1) %out, ptr addrspace(3) %in) #0 {
  %ld = load <16 x i8>, ptr addrspace(3) %in, align 1
  store <16 x i8> %ld, ptr addrspace(1) %out
  ret void
}

; SI-LABEL: {{^}}local_store_align1_v16i8:
; SI: ds_write_b8
; SI: ds_write_b8
; SI: ds_write_b8
; SI: ds_write_b8
; SI: ds_write_b8
; SI: ds_write_b8
; SI: ds_write_b8
; SI: ds_write_b8
; SI: ds_write_b8
; SI: ds_write_b8
; SI: ds_write_b8
; SI: ds_write_b8
; SI: ds_write_b8
; SI: ds_write_b8
; SI: ds_write_b8
; SI: ds_write_b8

; SI: ScratchSize: 0{{$}}
define amdgpu_kernel void @local_store_align1_v16i8(ptr addrspace(3) %out) #0 {
  store <16 x i8> zeroinitializer, ptr addrspace(3) %out, align 1
  ret void
}

; SI-LABEL: {{^}}private_load_align1_f64:
; MUBUF: buffer_load_ubyte
; MUBUF: buffer_load_ubyte
; MUBUF: buffer_load_ubyte
; MUBUF: buffer_load_ubyte
; MUBUF: buffer_load_ubyte
; MUBUF: buffer_load_ubyte
; MUBUF: buffer_load_ubyte
; MUBUF: buffer_load_ubyte
; FLATSCR: scratch_load_dwordx2
define double @private_load_align1_f64(ptr addrspace(5) %in) {
  %x = load double, ptr addrspace(5) %in, align 1
  ret double %x
}

; SI-LABEL: {{^}}private_store_align1_f64:
; MUBUF: buffer_store_byte
; MUBUF: buffer_store_byte
; MUBUF: buffer_store_byte
; MUBUF: buffer_store_byte
; MUBUF: buffer_store_byte
; MUBUF: buffer_store_byte
; MUBUF: buffer_store_byte
; MUBUF: buffer_store_byte
; FLATSCR: scratch_store_dwordx2
define void @private_store_align1_f64(ptr addrspace(5) %out, double %x) #0 {
  store double %x, ptr addrspace(5) %out, align 1
  ret void
}

; SI-LABEL: {{^}}private_load_align4_f64:
; MUBUF: buffer_load_dword
; MUBUF: buffer_load_dword
; FLATSCR: scratch_load_dwordx2
define double @private_load_align4_f64(ptr addrspace(5) %in) {
  %x = load double, ptr addrspace(5) %in, align 4
  ret double %x
}

; SI-LABEL: {{^}}private_store_align4_f64:
; MUBUF: buffer_store_dword
; MUBUF: buffer_store_dword
; FLATSCR: scratch_store_dwordx2
define void @private_store_align4_f64(ptr addrspace(5) %out, double %x) #0 {
  store double %x, ptr addrspace(5) %out, align 4
  ret void
}

; SI-LABEL: {{^}}private_load_align2_f64:
; MUBUF: buffer_load_ushort
; MUBUF: buffer_load_ushort
; MUBUF: buffer_load_ushort
; MUBUF: buffer_load_ushort
; FLATSCR: scratch_load_dwordx2
define double @private_load_align2_f64(ptr addrspace(5) %in) {
  %x = load double, ptr addrspace(5) %in, align 2
  ret double %x
}

; SI-LABEL: {{^}}private_store_align2_f64:
; MUBUF: buffer_store_short
; MUBUF: buffer_store_short
; MUBUF: buffer_store_short
; MUBUF: buffer_store_short
; FLATSCR: scratch_store_dwordx2
define void @private_store_align2_f64(ptr addrspace(5) %out, double %x) #0 {
  store double %x, ptr addrspace(5) %out, align 2
  ret void
}

; Should not merge this to a dword store
define amdgpu_kernel void @global_store_2xi16_align2(ptr addrspace(1) %p, ptr addrspace(1) %r) #0 {
  %gep.r = getelementptr i16, ptr addrspace(1) %r, i64 1
  %v = load i16, ptr addrspace(1) %p, align 2
  store i16 1, ptr addrspace(1) %r, align 2
  store i16 2, ptr addrspace(1) %gep.r, align 2
  ret void
}

; Should not merge this to a word load
define i32 @load_2xi16_align2(ptr addrspace(1) %p) #0 {
  %gep.p = getelementptr i16, ptr addrspace(1) %p, i64 1
  %p.0 = load i16, ptr addrspace(1) %p, align 2
  %p.1 = load i16, ptr addrspace(1) %gep.p, align 2
  %zext.0 = zext i16 %p.0 to i32
  %zext.1 = zext i16 %p.1 to i32
  %shl.1 = shl i32 %zext.1, 16
  %or = or i32 %zext.0, %shl.1
  ret i32 %or
}

attributes #0 = { nounwind }
