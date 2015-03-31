; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}global_copy_i1_to_i1:
; SI: buffer_load_ubyte
; SI: v_and_b32_e32 v{{[0-9]+}}, 1
; SI: buffer_store_byte
; SI: s_endpgm

; EG: VTX_READ_8
; EG: AND_INT
define void @global_copy_i1_to_i1(i1 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1 addrspace(1)* %in
  store i1 %load, i1 addrspace(1)* %out, align 1
  ret void
}

; FUNC-LABEL: {{^}}local_copy_i1_to_i1:
; SI: ds_read_u8
; SI: v_and_b32_e32 v{{[0-9]+}}, 1
; SI: ds_write_b8
; SI: s_endpgm

; EG: LDS_UBYTE_READ_RET
; EG: AND_INT
; EG: LDS_BYTE_WRITE
define void @local_copy_i1_to_i1(i1 addrspace(3)* %out, i1 addrspace(3)* %in) nounwind {
  %load = load i1 addrspace(3)* %in
  store i1 %load, i1 addrspace(3)* %out, align 1
  ret void
}

; FUNC-LABEL: {{^}}constant_copy_i1_to_i1:
; SI: buffer_load_ubyte
; SI: v_and_b32_e32 v{{[0-9]+}}, 1
; SI: buffer_store_byte
; SI: s_endpgm

; EG: VTX_READ_8
; EG: AND_INT
define void @constant_copy_i1_to_i1(i1 addrspace(1)* %out, i1 addrspace(2)* %in) nounwind {
  %load = load i1 addrspace(2)* %in
  store i1 %load, i1 addrspace(1)* %out, align 1
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_i1_to_i32:
; SI: buffer_load_ubyte
; SI: v_bfe_i32
; SI: buffer_store_dword
; SI: s_endpgm

; EG: VTX_READ_8
; EG: BFE_INT
define void @global_sextload_i1_to_i32(i32 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1 addrspace(1)* %in
  %ext = sext i1 %load to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_i1_to_i32:
; SI: buffer_load_ubyte
; SI: buffer_store_dword
; SI: s_endpgm

define void @global_zextload_i1_to_i32(i32 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1 addrspace(1)* %in
  %ext = zext i1 %load to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_i1_to_i64:
; SI: buffer_load_ubyte
; SI: v_bfe_i32
; SI: buffer_store_dwordx2
; SI: s_endpgm
define void @global_sextload_i1_to_i64(i64 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1 addrspace(1)* %in
  %ext = sext i1 %load to i64
  store i64 %ext, i64 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_i1_to_i64:
; SI: buffer_load_ubyte
; SI: v_mov_b32_e32 {{v[0-9]+}}, 0
; SI: buffer_store_dwordx2
; SI: s_endpgm
define void @global_zextload_i1_to_i64(i64 addrspace(1)* %out, i1 addrspace(1)* %in) nounwind {
  %load = load i1 addrspace(1)* %in
  %ext = zext i1 %load to i64
  store i64 %ext, i64 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i1_arg:
; SI: buffer_load_ubyte
; SI: v_and_b32_e32
; SI: buffer_store_byte
; SI: s_endpgm
define void @i1_arg(i1 addrspace(1)* %out, i1 %x) nounwind {
  store i1 %x, i1 addrspace(1)* %out, align 1
  ret void
}

; FUNC-LABEL: {{^}}i1_arg_zext_i32:
; SI: buffer_load_ubyte
; SI: buffer_store_dword
; SI: s_endpgm
define void @i1_arg_zext_i32(i32 addrspace(1)* %out, i1 %x) nounwind {
  %ext = zext i1 %x to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i1_arg_zext_i64:
; SI: buffer_load_ubyte
; SI: buffer_store_dwordx2
; SI: s_endpgm
define void @i1_arg_zext_i64(i64 addrspace(1)* %out, i1 %x) nounwind {
  %ext = zext i1 %x to i64
  store i64 %ext, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}i1_arg_sext_i32:
; SI: buffer_load_ubyte
; SI: buffer_store_dword
; SI: s_endpgm
define void @i1_arg_sext_i32(i32 addrspace(1)* %out, i1 %x) nounwind {
  %ext = sext i1 %x to i32
  store i32 %ext, i32addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i1_arg_sext_i64:
; SI: buffer_load_ubyte
; SI: v_bfe_i32
; SI: v_ashrrev_i32
; SI: buffer_store_dwordx2
; SI: s_endpgm
define void @i1_arg_sext_i64(i64 addrspace(1)* %out, i1 %x) nounwind {
  %ext = sext i1 %x to i64
  store i64 %ext, i64 addrspace(1)* %out, align 8
  ret void
}
