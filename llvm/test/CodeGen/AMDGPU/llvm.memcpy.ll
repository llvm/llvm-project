; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare void @llvm.memcpy.p3i8.p3i8.i32(i8 addrspace(3)* nocapture, i8 addrspace(3)* nocapture, i32, i1) nounwind
declare void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* nocapture, i8 addrspace(1)* nocapture, i64, i1) nounwind


; FUNC-LABEL: {{^}}test_small_memcpy_i64_lds_to_lds_align1:
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
define void @test_small_memcpy_i64_lds_to_lds_align1(i64 addrspace(3)* noalias %out, i64 addrspace(3)* noalias %in) nounwind {
  %bcin = bitcast i64 addrspace(3)* %in to i8 addrspace(3)*
  %bcout = bitcast i64 addrspace(3)* %out to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p3i8.i32(i8 addrspace(3)* align 1 %bcout, i8 addrspace(3)* align 1 %bcin, i32 32, i1 false) nounwind
  ret void
}

; FUNC-LABEL: {{^}}test_small_memcpy_i64_lds_to_lds_align2:
; SI: ds_read_u16
; SI: ds_read_u16
; SI: ds_read_u16
; SI: ds_read_u16
; SI: ds_read_u16
; SI: ds_read_u16
; SI: ds_read_u16
; SI: ds_read_u16

; SI: ds_read_u16
; SI: ds_read_u16
; SI: ds_read_u16
; SI: ds_read_u16
; SI: ds_read_u16
; SI: ds_read_u16
; SI: ds_read_u16
; SI: ds_read_u16

; SI: ds_write_b16
; SI: ds_write_b16
; SI: ds_write_b16
; SI: ds_write_b16
; SI: ds_write_b16
; SI: ds_write_b16
; SI: ds_write_b16
; SI: ds_write_b16

; SI: ds_write_b16
; SI: ds_write_b16
; SI: ds_write_b16
; SI: ds_write_b16
; SI: ds_write_b16
; SI: ds_write_b16
; SI: ds_write_b16
; SI: ds_write_b16

; SI: s_endpgm
define void @test_small_memcpy_i64_lds_to_lds_align2(i64 addrspace(3)* noalias %out, i64 addrspace(3)* noalias %in) nounwind {
  %bcin = bitcast i64 addrspace(3)* %in to i8 addrspace(3)*
  %bcout = bitcast i64 addrspace(3)* %out to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p3i8.i32(i8 addrspace(3)* align 2 %bcout, i8 addrspace(3)* align 2 %bcin, i32 32, i1 false) nounwind
  ret void
}

; FUNC-LABEL: {{^}}test_small_memcpy_i64_lds_to_lds_align4:
; SI-DAG: ds_read_b32
; SI-DAG: ds_write_b32

; SI-DAG: ds_read_b32
; SI-DAG: ds_write_b32

; SI-DAG: ds_read_b32
; SI-DAG: ds_write_b32

; SI-DAG: ds_read_b32
; SI-DAG: ds_write_b32

; SI-DAG: ds_read_b32
; SI-DAG: ds_write_b32

; SI-DAG: ds_read_b32
; SI-DAG: ds_write_b32

; SI-DAG: ds_read_b32
; SI-DAG: ds_write_b32

; SI-DAG: ds_read_b32
; SI-DAG: ds_write_b32

; SI-DAG: ds_read_b32
; SI-DAG: ds_write_b32

; SI: s_endpgm
define void @test_small_memcpy_i64_lds_to_lds_align4(i64 addrspace(3)* noalias %out, i64 addrspace(3)* noalias %in) nounwind {
  %bcin = bitcast i64 addrspace(3)* %in to i8 addrspace(3)*
  %bcout = bitcast i64 addrspace(3)* %out to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p3i8.i32(i8 addrspace(3)* align 4 %bcout, i8 addrspace(3)* align 4 %bcin, i32 32, i1 false) nounwind
  ret void
}

; FIXME: Use 64-bit ops
; FUNC-LABEL: {{^}}test_small_memcpy_i64_lds_to_lds_align8:

; SI-DAG: ds_read_b32
; SI-DAG: ds_write_b32

; SI-DAG: ds_read_b32
; SI-DAG: ds_write_b32

; SI-DAG: ds_read_b32
; SI-DAG: ds_write_b32

; SI-DAG: ds_read_b32
; SI-DAG: ds_write_b32

; SI-DAG: ds_read_b32
; SI-DAG: ds_write_b32

; SI-DAG: ds_read_b32
; SI-DAG: ds_write_b32

; SI-DAG: ds_read_b32
; SI-DAG: ds_write_b32

; SI-DAG: ds_read_b32
; SI-DAG: ds_write_b32

; SI-DAG: ds_read_b32
; SI-DAG: ds_write_b32

; SI-DAG: s_endpgm
define void @test_small_memcpy_i64_lds_to_lds_align8(i64 addrspace(3)* noalias %out, i64 addrspace(3)* noalias %in) nounwind {
  %bcin = bitcast i64 addrspace(3)* %in to i8 addrspace(3)*
  %bcout = bitcast i64 addrspace(3)* %out to i8 addrspace(3)*
  call void @llvm.memcpy.p3i8.p3i8.i32(i8 addrspace(3)* align 8 %bcout, i8 addrspace(3)* align 8 %bcin, i32 32, i1 false) nounwind
  ret void
}

; FUNC-LABEL: {{^}}test_small_memcpy_i64_global_to_global_align1:
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte

; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte

; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte

; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte
; SI-DAG: buffer_load_ubyte
; SI-DAG: buffer_store_byte

; SI: s_endpgm
define void @test_small_memcpy_i64_global_to_global_align1(i64 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %in) nounwind {
  %bcin = bitcast i64 addrspace(1)* %in to i8 addrspace(1)*
  %bcout = bitcast i64 addrspace(1)* %out to i8 addrspace(1)*
  call void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* align 1 %bcout, i8 addrspace(1)* align 1 %bcin, i64 32, i1 false) nounwind
  ret void
}

; FUNC-LABEL: {{^}}test_small_memcpy_i64_global_to_global_align2:
; SI-DAG: buffer_load_ushort
; SI-DAG: buffer_load_ushort
; SI-DAG: buffer_load_ushort
; SI-DAG: buffer_load_ushort
; SI-DAG: buffer_load_ushort
; SI-DAG: buffer_load_ushort
; SI-DAG: buffer_load_ushort
; SI-DAG: buffer_load_ushort
; SI-DAG: buffer_load_ushort
; SI-DAG: buffer_load_ushort
; SI-DAG: buffer_load_ushort
; SI-DAG: buffer_load_ushort
; SI-DAG: buffer_load_ushort
; SI-DAG: buffer_load_ushort
; SI-DAG: buffer_load_ushort
; SI-DAG: buffer_load_ushort

; SI-DAG: buffer_store_short
; SI-DAG: buffer_store_short
; SI-DAG: buffer_store_short
; SI-DAG: buffer_store_short
; SI-DAG: buffer_store_short
; SI-DAG: buffer_store_short
; SI-DAG: buffer_store_short
; SI-DAG: buffer_store_short
; SI-DAG: buffer_store_short
; SI-DAG: buffer_store_short
; SI-DAG: buffer_store_short
; SI-DAG: buffer_store_short
; SI-DAG: buffer_store_short
; SI-DAG: buffer_store_short
; SI-DAG: buffer_store_short
; SI-DAG: buffer_store_short

; SI: s_endpgm
define void @test_small_memcpy_i64_global_to_global_align2(i64 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %in) nounwind {
  %bcin = bitcast i64 addrspace(1)* %in to i8 addrspace(1)*
  %bcout = bitcast i64 addrspace(1)* %out to i8 addrspace(1)*
  call void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* align 2 %bcout, i8 addrspace(1)* align 2 %bcin, i64 32, i1 false) nounwind
  ret void
}

; FUNC-LABEL: {{^}}test_small_memcpy_i64_global_to_global_align4:
; SI: buffer_load_dwordx4
; SI: buffer_load_dwordx4
; SI: buffer_store_dwordx4
; SI: buffer_store_dwordx4
; SI: s_endpgm
define void @test_small_memcpy_i64_global_to_global_align4(i64 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %in) nounwind {
  %bcin = bitcast i64 addrspace(1)* %in to i8 addrspace(1)*
  %bcout = bitcast i64 addrspace(1)* %out to i8 addrspace(1)*
  call void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* align 4 %bcout, i8 addrspace(1)* align 4 %bcin, i64 32, i1 false) nounwind
  ret void
}

; FUNC-LABEL: {{^}}test_small_memcpy_i64_global_to_global_align8:
; SI: buffer_load_dwordx4
; SI: buffer_load_dwordx4
; SI: buffer_store_dwordx4
; SI: buffer_store_dwordx4
; SI: s_endpgm
define void @test_small_memcpy_i64_global_to_global_align8(i64 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %in) nounwind {
  %bcin = bitcast i64 addrspace(1)* %in to i8 addrspace(1)*
  %bcout = bitcast i64 addrspace(1)* %out to i8 addrspace(1)*
  call void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* align 8 %bcout, i8 addrspace(1)* align 8 %bcin, i64 32, i1 false) nounwind
  ret void
}

; FUNC-LABEL: {{^}}test_small_memcpy_i64_global_to_global_align16:
; SI: buffer_load_dwordx4
; SI: buffer_load_dwordx4
; SI: buffer_store_dwordx4
; SI: buffer_store_dwordx4
; SI: s_endpgm
define void @test_small_memcpy_i64_global_to_global_align16(i64 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %in) nounwind {
  %bcin = bitcast i64 addrspace(1)* %in to i8 addrspace(1)*
  %bcout = bitcast i64 addrspace(1)* %out to i8 addrspace(1)*
  call void @llvm.memcpy.p1i8.p1i8.i64(i8 addrspace(1)* align 16 %bcout, i8 addrspace(1)* align 16 %bcin, i64 32, i1 false) nounwind
  ret void
}
