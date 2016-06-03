; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-NOHSA -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-HSA -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-NOHSA -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}local_unaligned_load_store_i16:
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: s_endpgm
define void @local_unaligned_load_store_i16(i16 addrspace(3)* %p, i16 addrspace(3)* %r) #0 {
  %v = load i16, i16 addrspace(3)* %p, align 1
  store i16 %v, i16 addrspace(3)* %r, align 1
  ret void
}

; FUNC-LABEL: {{^}}unaligned_load_store_i16_global:
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte

; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
define void @unaligned_load_store_i16_global(i16 addrspace(1)* %p, i16 addrspace(1)* %r) #0 {
  %v = load i16, i16 addrspace(1)* %p, align 1
  store i16 %v, i16 addrspace(1)* %r, align 1
  ret void
}

; FUNC-LABEL: {{^}}local_unaligned_load_store_i32:
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: s_endpgm
define void @local_unaligned_load_store_i32(i32 addrspace(3)* %p, i32 addrspace(3)* %r) #0 {
  %v = load i32, i32 addrspace(3)* %p, align 1
  store i32 %v, i32 addrspace(3)* %r, align 1
  ret void
}

; FUNC-LABEL: {{^}}global_unaligned_load_store_i32:
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte

; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
define void @global_unaligned_load_store_i32(i32 addrspace(1)* %p, i32 addrspace(1)* %r) #0 {
  %v = load i32, i32 addrspace(1)* %p, align 1
  store i32 %v, i32 addrspace(1)* %r, align 1
  ret void
}

; FUNC-LABEL: {{^}}global_align2_load_store_i32:
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_store_short
; GCN-NOHSA: buffer_store_short

; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_store_short
; GCN-HSA: flat_store_short
define void @global_align2_load_store_i32(i32 addrspace(1)* %p, i32 addrspace(1)* %r) #0 {
  %v = load i32, i32 addrspace(1)* %p, align 2
  store i32 %v, i32 addrspace(1)* %r, align 2
  ret void
}

; FUNC-LABEL: {{^}}local_align2_load_store_i32:
; GCN: ds_read_u16
; GCN: ds_read_u16
; GCN: ds_write_b16
; GCN: ds_write_b16
define void @local_align2_load_store_i32(i32 addrspace(3)* %p, i32 addrspace(3)* %r) #0 {
  %v = load i32, i32 addrspace(3)* %p, align 2
  store i32 %v, i32 addrspace(3)* %r, align 2
  ret void
}

; FIXME: Unnecessary packing and unpacking of bytes.
; FUNC-LABEL: {{^}}local_unaligned_load_store_i64:
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8

; XGCN-NOT: v_or_b32
; XGCN-NOT: v_lshl
; GCN: ds_write_b8
; XGCN-NOT: v_or_b32
; XGCN-NOT: v_lshl

; GCN: ds_write_b8
; XGCN-NOT: v_or_b32
; XGCN-NOT: v_lshl

; GCN: ds_write_b8
; XGCN-NOT: v_or_b32
; XGCN-NOT: v_lshl

; GCN: ds_write_b8
; XGCN-NOT: v_or_b32
; XGCN-NOT: v_lshl

; GCN: ds_write_b8
; XGCN-NOT: v_or_b32
; XGCN-NOT: v_lshl

; GCN: ds_write_b8
; XGCN-NOT: v_or_b32
; XGCN-NOT: v_lshl

; GCN: ds_write_b8
; XGCN-NOT: v_or_b32
; XGCN-NOT: v_lshl
; GCN: ds_write_b8
; GCN: s_endpgm
define void @local_unaligned_load_store_i64(i64 addrspace(3)* %p, i64 addrspace(3)* %r) {
  %v = load i64, i64 addrspace(3)* %p, align 1
  store i64 %v, i64 addrspace(3)* %r, align 1
  ret void
}

; FUNC-LABEL: {{^}}local_unaligned_load_store_v2i32:
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8

; XGCN-NOT: v_or_b32
; XGCN-NOT: v_lshl
; GCN: ds_write_b8
; XGCN-NOT: v_or_b32
; XGCN-NOT: v_lshl

; GCN: ds_write_b8
; XGCN-NOT: v_or_b32
; XGCN-NOT: v_lshl

; GCN: ds_write_b8
; XGCN-NOT: v_or_b32
; XGCN-NOT: v_lshl

; GCN: ds_write_b8
; XGCN-NOT: v_or_b32
; XGCN-NOT: v_lshl

; GCN: ds_write_b8
; XGCN-NOT: v_or_b32
; XGCN-NOT: v_lshl

; GCN: ds_write_b8
; XGCN-NOT: v_or_b32
; XGCN-NOT: v_lshl

; GCN: ds_write_b8
; XGCN-NOT: v_or_b32
; XGCN-NOT: v_lshl
; GCN: ds_write_b8
; GCN: s_endpgm
define void @local_unaligned_load_store_v2i32(<2 x i32> addrspace(3)* %p, <2 x i32> addrspace(3)* %r) {
  %v = load <2 x i32>, <2 x i32> addrspace(3)* %p, align 1
  store <2 x i32> %v, <2 x i32> addrspace(3)* %r, align 1
  ret void
}

; FUNC-LABEL: {{^}}unaligned_load_store_i64_global:
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte

; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte

; XGCN-NOT: v_or_
; XGCN-NOT: v_lshl

; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte

; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
define void @unaligned_load_store_i64_global(i64 addrspace(1)* %p, i64 addrspace(1)* %r) #0 {
  %v = load i64, i64 addrspace(1)* %p, align 1
  store i64 %v, i64 addrspace(1)* %r, align 1
  ret void
}

; FUNC-LABEL: {{^}}local_unaligned_load_store_v4i32:
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
define void @local_unaligned_load_store_v4i32(<4 x i32> addrspace(3)* %p, <4 x i32> addrspace(3)* %r) #0 {
  %v = load <4 x i32>, <4 x i32> addrspace(3)* %p, align 1
  store <4 x i32> %v, <4 x i32> addrspace(3)* %r, align 1
  ret void
}

; FUNC-LABEL: {{^}}global_unaligned_load_store_v4i32:
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte

; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte
; GCN-NOHSA: buffer_store_byte


; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte

; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
; GCN-HSA: flat_store_byte
define void @global_unaligned_load_store_v4i32(<4 x i32> addrspace(1)* %p, <4 x i32> addrspace(1)* %r) #0 {
  %v = load <4 x i32>, <4 x i32> addrspace(1)* %p, align 1
  store <4 x i32> %v, <4 x i32> addrspace(1)* %r, align 1
  ret void
}

; FUNC-LABEL: {{^}}local_load_i64_align_4:
; GCN: ds_read2_b32
define void @local_load_i64_align_4(i64 addrspace(1)* nocapture %out, i64 addrspace(3)* %in) #0 {
  %val = load i64, i64 addrspace(3)* %in, align 4
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}local_load_i64_align_4_with_offset
; GCN: ds_read2_b32 v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]}} offset0:8 offset1:9
define void @local_load_i64_align_4_with_offset(i64 addrspace(1)* nocapture %out, i64 addrspace(3)* %in) #0 {
  %ptr = getelementptr i64, i64 addrspace(3)* %in, i32 4
  %val = load i64, i64 addrspace(3)* %ptr, align 4
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}local_load_i64_align_4_with_split_offset:
; The tests for the case where the lo offset is 8-bits, but the hi offset is 9-bits
; GCN: ds_read2_b32 v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]}} offset1:1
; GCN: s_endpgm
define void @local_load_i64_align_4_with_split_offset(i64 addrspace(1)* nocapture %out, i64 addrspace(3)* %in) #0 {
  %ptr = bitcast i64 addrspace(3)* %in to i32 addrspace(3)*
  %ptr255 = getelementptr i32, i32 addrspace(3)* %ptr, i32 255
  %ptri64 = bitcast i32 addrspace(3)* %ptr255 to i64 addrspace(3)*
  %val = load i64, i64 addrspace(3)* %ptri64, align 4
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}local_load_i64_align_1:
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: ds_read_u8
; GCN: store_dwordx2
define void @local_load_i64_align_1(i64 addrspace(1)* nocapture %out, i64 addrspace(3)* %in) #0 {
  %val = load i64, i64 addrspace(3)* %in, align 1
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}local_store_i64_align_4:
; GCN: ds_write2_b32
define void @local_store_i64_align_4(i64 addrspace(3)* %out, i64 %val) #0 {
  store i64 %val, i64 addrspace(3)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}local_store_i64_align_4_with_offset
; GCN: ds_write2_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset0:8 offset1:9
; GCN: s_endpgm
define void @local_store_i64_align_4_with_offset(i64 addrspace(3)* %out) #0 {
  %ptr = getelementptr i64, i64 addrspace(3)* %out, i32 4
  store i64 0, i64 addrspace(3)* %ptr, align 4
  ret void
}

; FUNC-LABEL: {{^}}local_store_i64_align_4_with_split_offset:
; The tests for the case where the lo offset is 8-bits, but the hi offset is 9-bits
; GCN: ds_write2_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset1:1
; GCN: s_endpgm
define void @local_store_i64_align_4_with_split_offset(i64 addrspace(3)* %out) #0 {
  %ptr = bitcast i64 addrspace(3)* %out to i32 addrspace(3)*
  %ptr255 = getelementptr i32, i32 addrspace(3)* %ptr, i32 255
  %ptri64 = bitcast i32 addrspace(3)* %ptr255 to i64 addrspace(3)*
  store i64 0, i64 addrspace(3)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}constant_load_unaligned_i16:
; GCN-NOHSA: buffer_load_ushort
; GCN-HSA: flat_load_ushort

; EG: VTX_READ_16 T{{[0-9]+\.X, T[0-9]+\.X}}
define void @constant_load_unaligned_i16(i32 addrspace(1)* %out, i16 addrspace(2)* %in) {
entry:
  %tmp0 = getelementptr i16, i16 addrspace(2)* %in, i32 1
  %tmp1 = load i16, i16 addrspace(2)* %tmp0
  %tmp2 = zext i16 %tmp1 to i32
  store i32 %tmp2, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_load_unaligned_i32:
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte

; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
define void @constant_load_unaligned_i32(i32 addrspace(1)* %out, i32 addrspace(2)* %in) {
entry:
  %tmp0 = load i32, i32 addrspace(2)* %in, align 1
  store i32 %tmp0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_load_unaligned_f32:
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte
; GCN-NOHSA: buffer_load_ubyte

; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
; GCN-HSA: flat_load_ubyte
define void @constant_load_unaligned_f32(float addrspace(1)* %out, float addrspace(2)* %in) {
  %tmp1 = load float, float addrspace(2)* %in, align 1
  store float %tmp1, float addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
