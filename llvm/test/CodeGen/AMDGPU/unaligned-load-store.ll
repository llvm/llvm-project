; RUN: llc -march=amdgcn -verify-machineinstrs< %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs< %s | FileCheck -check-prefix=SI %s

; SI-LABEL: {{^}}unaligned_load_store_i16_local:
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_write_b8
; SI: ds_write_b8
; SI: s_endpgm
define void @unaligned_load_store_i16_local(i16 addrspace(3)* %p, i16 addrspace(3)* %r) nounwind {
  %v = load i16, i16 addrspace(3)* %p, align 1
  store i16 %v, i16 addrspace(3)* %r, align 1
  ret void
}

; SI-LABEL: {{^}}unaligned_load_store_i16_global:
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: s_endpgm
define void @unaligned_load_store_i16_global(i16 addrspace(1)* %p, i16 addrspace(1)* %r) nounwind {
  %v = load i16, i16 addrspace(1)* %p, align 1
  store i16 %v, i16 addrspace(1)* %r, align 1
  ret void
}

; SI-LABEL: {{^}}unaligned_load_store_i32_local:
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_write_b8
; SI: ds_write_b8
; SI: ds_write_b8
; SI: ds_write_b8
; SI: s_endpgm
define void @unaligned_load_store_i32_local(i32 addrspace(3)* %p, i32 addrspace(3)* %r) nounwind {
  %v = load i32, i32 addrspace(3)* %p, align 1
  store i32 %v, i32 addrspace(3)* %r, align 1
  ret void
}

; SI-LABEL: {{^}}unaligned_load_store_i32_global:
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
define void @unaligned_load_store_i32_global(i32 addrspace(1)* %p, i32 addrspace(1)* %r) nounwind {
  %v = load i32, i32 addrspace(1)* %p, align 1
  store i32 %v, i32 addrspace(1)* %r, align 1
  ret void
}

; SI-LABEL: {{^}}align2_load_store_i32_global:
; SI: buffer_load_ushort
; SI: buffer_load_ushort
; SI: buffer_store_short
; SI: buffer_store_short
define void @align2_load_store_i32_global(i32 addrspace(1)* %p, i32 addrspace(1)* %r) nounwind {
  %v = load i32, i32 addrspace(1)* %p, align 2
  store i32 %v, i32 addrspace(1)* %r, align 2
  ret void
}

; SI-LABEL: {{^}}align2_load_store_i32_local:
; SI: ds_read_u16
; SI: ds_read_u16
; SI: ds_write_b16
; SI: ds_write_b16
define void @align2_load_store_i32_local(i32 addrspace(3)* %p, i32 addrspace(3)* %r) nounwind {
  %v = load i32, i32 addrspace(3)* %p, align 2
  store i32 %v, i32 addrspace(3)* %r, align 2
  ret void
}

; FIXME: Unnecessary packing and unpacking of bytes.
; SI-LABEL: {{^}}unaligned_load_store_i64_local:
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8

; XSI-NOT: v_or_b32
; XSI-NOT: v_lshl
; SI: ds_write_b8
; XSI-NOT: v_or_b32
; XSI-NOT: v_lshl

; SI: ds_write_b8
; XSI-NOT: v_or_b32
; XSI-NOT: v_lshl

; SI: ds_write_b8
; XSI-NOT: v_or_b32
; XSI-NOT: v_lshl

; SI: ds_write_b8
; XSI-NOT: v_or_b32
; XSI-NOT: v_lshl

; SI: ds_write_b8
; XSI-NOT: v_or_b32
; XSI-NOT: v_lshl

; SI: ds_write_b8
; XSI-NOT: v_or_b32
; XSI-NOT: v_lshl

; SI: ds_write_b8
; XSI-NOT: v_or_b32
; XSI-NOT: v_lshl
; SI: ds_write_b8
; SI: s_endpgm
define void @unaligned_load_store_i64_local(i64 addrspace(3)* %p, i64 addrspace(3)* %r) {
  %v = load i64, i64 addrspace(3)* %p, align 1
  store i64 %v, i64 addrspace(3)* %r, align 1
  ret void
}

; SI-LABEL: {{^}}unaligned_load_store_v2i32_local:
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8

; XSI-NOT: v_or_b32
; XSI-NOT: v_lshl
; SI: ds_write_b8
; XSI-NOT: v_or_b32
; XSI-NOT: v_lshl

; SI: ds_write_b8
; XSI-NOT: v_or_b32
; XSI-NOT: v_lshl

; SI: ds_write_b8
; XSI-NOT: v_or_b32
; XSI-NOT: v_lshl

; SI: ds_write_b8
; XSI-NOT: v_or_b32
; XSI-NOT: v_lshl

; SI: ds_write_b8
; XSI-NOT: v_or_b32
; XSI-NOT: v_lshl

; SI: ds_write_b8
; XSI-NOT: v_or_b32
; XSI-NOT: v_lshl

; SI: ds_write_b8
; XSI-NOT: v_or_b32
; XSI-NOT: v_lshl
; SI: ds_write_b8
; SI: s_endpgm
define void @unaligned_load_store_v2i32_local(<2 x i32> addrspace(3)* %p, <2 x i32> addrspace(3)* %r) {
  %v = load <2 x i32>, <2 x i32> addrspace(3)* %p, align 1
  store <2 x i32> %v, <2 x i32> addrspace(3)* %r, align 1
  ret void
}

; SI-LABEL: {{^}}unaligned_load_store_i64_global:
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte

; XSI-NOT: v_or_
; XSI-NOT: v_lshl

; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
define void @unaligned_load_store_i64_global(i64 addrspace(1)* %p, i64 addrspace(1)* %r) {
  %v = load i64, i64 addrspace(1)* %p, align 1
  store i64 %v, i64 addrspace(1)* %r, align 1
  ret void
}

; SI-LABEL: {{^}}unaligned_load_store_v4i32_local:
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
; SI: s_endpgm
define void @unaligned_load_store_v4i32_local(<4 x i32> addrspace(3)* %p, <4 x i32> addrspace(3)* %r) nounwind {
  %v = load <4 x i32>, <4 x i32> addrspace(3)* %p, align 1
  store <4 x i32> %v, <4 x i32> addrspace(3)* %r, align 1
  ret void
}

; FIXME: We mark v4i32 as custom, so misaligned loads are never expanded.
; FIXME-SI-LABEL: {{^}}unaligned_load_store_v4i32_global
; FIXME-SI: buffer_load_ubyte
; FIXME-SI: buffer_load_ubyte
; FIXME-SI: buffer_load_ubyte
; FIXME-SI: buffer_load_ubyte
; FIXME-SI: buffer_load_ubyte
; FIXME-SI: buffer_load_ubyte
; FIXME-SI: buffer_load_ubyte
; FIXME-SI: buffer_load_ubyte
; FIXME-SI: buffer_load_ubyte
; FIXME-SI: buffer_load_ubyte
; FIXME-SI: buffer_load_ubyte
; FIXME-SI: buffer_load_ubyte
; FIXME-SI: buffer_load_ubyte
; FIXME-SI: buffer_load_ubyte
; FIXME-SI: buffer_load_ubyte
; FIXME-SI: buffer_load_ubyte
define void @unaligned_load_store_v4i32_global(<4 x i32> addrspace(1)* %p, <4 x i32> addrspace(1)* %r) nounwind {
  %v = load <4 x i32>, <4 x i32> addrspace(1)* %p, align 1
  store <4 x i32> %v, <4 x i32> addrspace(1)* %r, align 1
  ret void
}

; SI-LABEL: {{^}}load_lds_i64_align_4:
; SI: ds_read2_b32
; SI: s_endpgm
define void @load_lds_i64_align_4(i64 addrspace(1)* nocapture %out, i64 addrspace(3)* %in) #0 {
  %val = load i64, i64 addrspace(3)* %in, align 4
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}load_lds_i64_align_4_with_offset
; SI: ds_read2_b32 v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]}} offset0:8 offset1:9
; SI: s_endpgm
define void @load_lds_i64_align_4_with_offset(i64 addrspace(1)* nocapture %out, i64 addrspace(3)* %in) #0 {
  %ptr = getelementptr i64, i64 addrspace(3)* %in, i32 4
  %val = load i64, i64 addrspace(3)* %ptr, align 4
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}load_lds_i64_align_4_with_split_offset:
; The tests for the case where the lo offset is 8-bits, but the hi offset is 9-bits
; SI: ds_read2_b32 v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]}} offset1:1
; SI: s_endpgm
define void @load_lds_i64_align_4_with_split_offset(i64 addrspace(1)* nocapture %out, i64 addrspace(3)* %in) #0 {
  %ptr = bitcast i64 addrspace(3)* %in to i32 addrspace(3)*
  %ptr255 = getelementptr i32, i32 addrspace(3)* %ptr, i32 255
  %ptri64 = bitcast i32 addrspace(3)* %ptr255 to i64 addrspace(3)*
  %val = load i64, i64 addrspace(3)* %ptri64, align 4
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}load_lds_i64_align_1:
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: ds_read_u8
; SI: buffer_store_dwordx2
; SI: s_endpgm

define void @load_lds_i64_align_1(i64 addrspace(1)* nocapture %out, i64 addrspace(3)* %in) #0 {
  %val = load i64, i64 addrspace(3)* %in, align 1
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}store_lds_i64_align_4:
; SI: ds_write2_b32
; SI: s_endpgm
define void @store_lds_i64_align_4(i64 addrspace(3)* %out, i64 %val) #0 {
  store i64 %val, i64 addrspace(3)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}store_lds_i64_align_4_with_offset
; SI: ds_write2_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset0:8 offset1:9
; SI: s_endpgm
define void @store_lds_i64_align_4_with_offset(i64 addrspace(3)* %out) #0 {
  %ptr = getelementptr i64, i64 addrspace(3)* %out, i32 4
  store i64 0, i64 addrspace(3)* %ptr, align 4
  ret void
}

; SI-LABEL: {{^}}store_lds_i64_align_4_with_split_offset:
; The tests for the case where the lo offset is 8-bits, but the hi offset is 9-bits
; SI: ds_write2_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset1:1
; SI: s_endpgm
define void @store_lds_i64_align_4_with_split_offset(i64 addrspace(3)* %out) #0 {
  %ptr = bitcast i64 addrspace(3)* %out to i32 addrspace(3)*
  %ptr255 = getelementptr i32, i32 addrspace(3)* %ptr, i32 255
  %ptri64 = bitcast i32 addrspace(3)* %ptr255 to i64 addrspace(3)*
  store i64 0, i64 addrspace(3)* %out, align 4
  ret void
}
