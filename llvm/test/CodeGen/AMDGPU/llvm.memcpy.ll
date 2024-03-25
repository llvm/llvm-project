; RUN: llc -mtriple=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare void @llvm.memcpy.p3.p3.i32(ptr addrspace(3) nocapture, ptr addrspace(3) nocapture, i32, i1) nounwind
declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) nocapture, ptr addrspace(1) nocapture, i64, i1) nounwind
declare void @llvm.memcpy.p1.p2.i64(ptr addrspace(1) nocapture, ptr addrspace(4) nocapture, i64, i1) nounwind


; FUNC-LABEL: {{^}}test_small_memcpy_i64_lds_to_lds_align1:
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8

; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8

; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8

; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8
; SI-DAG: ds_read_u8

; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8

; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8

; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8

; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8
; SI-DAG: ds_write_b8

; SI: s_endpgm
define amdgpu_kernel void @test_small_memcpy_i64_lds_to_lds_align1(ptr addrspace(3) noalias %out, ptr addrspace(3) noalias %in) nounwind {
  call void @llvm.memcpy.p3.p3.i32(ptr addrspace(3) %out, ptr addrspace(3) %in, i32 32, i1 false) nounwind
  ret void
}

; FUNC-LABEL: {{^}}test_small_memcpy_i64_lds_to_lds_align2:
; SI-DAG: ds_read_u16
; SI-DAG: ds_read_u16
; SI-DAG: ds_read_u16
; SI-DAG: ds_read_u16
; SI-DAG: ds_read_u16
; SI-DAG: ds_read_u16
; SI-DAG: ds_read_u16
; SI-DAG: ds_read_u16

; SI-DAG: ds_read_u16
; SI-DAG: ds_read_u16
; SI-DAG: ds_read_u16
; SI-DAG: ds_read_u16
; SI-DAG: ds_read_u16
; SI-DAG: ds_read_u16
; SI-DAG: ds_read_u16
; SI-DAG: ds_read_u16

; SI-DAG: ds_write_b16
; SI-DAG: ds_write_b16
; SI-DAG: ds_write_b16
; SI-DAG: ds_write_b16
; SI-DAG: ds_write_b16
; SI-DAG: ds_write_b16
; SI-DAG: ds_write_b16
; SI-DAG: ds_write_b16

; SI-DAG: ds_write_b16
; SI-DAG: ds_write_b16
; SI-DAG: ds_write_b16
; SI-DAG: ds_write_b16
; SI-DAG: ds_write_b16
; SI-DAG: ds_write_b16
; SI-DAG: ds_write_b16
; SI-DAG: ds_write_b16

; SI: s_endpgm
define amdgpu_kernel void @test_small_memcpy_i64_lds_to_lds_align2(ptr addrspace(3) noalias %out, ptr addrspace(3) noalias %in) nounwind {
  call void @llvm.memcpy.p3.p3.i32(ptr addrspace(3) align 2 %out, ptr addrspace(3) align 2 %in, i32 32, i1 false) nounwind
  ret void
}

; FUNC-LABEL: {{^}}test_small_memcpy_i64_lds_to_lds_align4:
; SI: ds_read2_b32
; SI: ds_read2_b32
; SI: ds_read2_b32
; SI: ds_read2_b32

; SI: ds_write2_b32
; SI: ds_write2_b32
; SI: ds_write2_b32
; SI: ds_write2_b32

; SI: s_endpgm
define amdgpu_kernel void @test_small_memcpy_i64_lds_to_lds_align4(ptr addrspace(3) noalias %out, ptr addrspace(3) noalias %in) nounwind {
  call void @llvm.memcpy.p3.p3.i32(ptr addrspace(3) align 4 %out, ptr addrspace(3) align 4 %in, i32 32, i1 false) nounwind
  ret void
}

; FIXME: Use 64-bit ops
; FUNC-LABEL: {{^}}test_small_memcpy_i64_lds_to_lds_align8:

; SI: ds_read2_b64
; SI: ds_read2_b64

; SI: ds_write2_b64
; SI: ds_write2_b64

; SI-DAG: s_endpgm
define amdgpu_kernel void @test_small_memcpy_i64_lds_to_lds_align8(ptr addrspace(3) noalias %out, ptr addrspace(3) noalias %in) nounwind {
  call void @llvm.memcpy.p3.p3.i32(ptr addrspace(3) align 8 %out, ptr addrspace(3) align 8 %in, i32 32, i1 false) nounwind
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
define amdgpu_kernel void @test_small_memcpy_i64_global_to_global_align1(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in) nounwind {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %out, ptr addrspace(1) %in, i64 32, i1 false) nounwind
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
define amdgpu_kernel void @test_small_memcpy_i64_global_to_global_align2(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in) nounwind {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 2 %out, ptr addrspace(1) align 2 %in, i64 32, i1 false) nounwind
  ret void
}

; FUNC-LABEL: {{^}}test_small_memcpy_i64_global_to_global_align4:
; SI: buffer_load_dwordx4
; SI: buffer_load_dwordx4
; SI: buffer_store_dwordx4
; SI: buffer_store_dwordx4
; SI: s_endpgm
define amdgpu_kernel void @test_small_memcpy_i64_global_to_global_align4(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in) nounwind {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 4 %out, ptr addrspace(1) align 4 %in, i64 32, i1 false) nounwind
  ret void
}

; FUNC-LABEL: {{^}}test_small_memcpy_i64_global_to_global_align8:
; SI: buffer_load_dwordx4
; SI: buffer_load_dwordx4
; SI: buffer_store_dwordx4
; SI: buffer_store_dwordx4
; SI: s_endpgm
define amdgpu_kernel void @test_small_memcpy_i64_global_to_global_align8(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in) nounwind {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 8 %out, ptr addrspace(1) align 8 %in, i64 32, i1 false) nounwind
  ret void
}

; FUNC-LABEL: {{^}}test_small_memcpy_i64_global_to_global_align16:
; SI: buffer_load_dwordx4
; SI: buffer_load_dwordx4
; SI: buffer_store_dwordx4
; SI: buffer_store_dwordx4
; SI: s_endpgm
define amdgpu_kernel void @test_small_memcpy_i64_global_to_global_align16(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in) nounwind {
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 16 %out, ptr addrspace(1) align 16 %in, i64 32, i1 false) nounwind
  ret void
}

; Test shouldConvertConstantLoadToIntImm
@hello.align4 = private unnamed_addr addrspace(4) constant [16 x i8] c"constant string\00", align 4
@hello.align1 = private unnamed_addr addrspace(4) constant [16 x i8] c"constant string\00", align 1

; FUNC-LABEL: {{^}}test_memcpy_const_string_align4:
; SI: s_getpc_b64
; SI: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, hello.align4@rel32@lo+4
; SI: s_addc_u32
; SI-DAG: s_load_dwordx8
; SI-DAG: s_load_dwordx2
; SI-DAG: buffer_store_dwordx4
; SI-DAG: buffer_store_dwordx4
define amdgpu_kernel void @test_memcpy_const_string_align4(ptr addrspace(1) noalias %out) nounwind {
  call void @llvm.memcpy.p1.p2.i64(ptr addrspace(1) align 4 %out, ptr addrspace(4) align 4 @hello.align4, i64 32, i1 false)
  ret void
}

; FUNC-LABEL: {{^}}test_memcpy_const_string_align1:
; SI-NOT: buffer_load
; SI: v_mov_b32_e32 v{{[0-9]+}}, 0x
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
define amdgpu_kernel void @test_memcpy_const_string_align1(ptr addrspace(1) noalias %out) nounwind {
  call void @llvm.memcpy.p1.p2.i64(ptr addrspace(1) %out, ptr addrspace(4) @hello.align1, i64 32, i1 false)
  ret void
}
