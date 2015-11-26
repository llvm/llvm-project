; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN -check-prefix=GCN-NOAA %s
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN -check-prefix=GCN-NOAA %s

; RUN: llc -march=amdgcn -verify-machineinstrs -combiner-alias-analysis < %s | FileCheck -check-prefix=SI -check-prefix=GCN -check-prefix=GCN-AA %s
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs -combiner-alias-analysis < %s | FileCheck -check-prefix=SI -check-prefix=GCN -check-prefix=GCN-AA %s

; Run with devices with different unaligned load restrictions.

; TODO: Vector element tests
; TODO: Non-zero base offset for load and store combinations
; TODO: Same base addrspacecasted


; GCN-LABEL: {{^}}merge_global_store_2_constants_i8:
; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: s_endpgm
define void @merge_global_store_2_constants_i8(i8 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i8, i8 addrspace(1)* %out, i32 1

  store i8 123, i8 addrspace(1)* %out.gep.1
  store i8 456, i8 addrspace(1)* %out, align 2
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_2_constants_i8_natural_align:
; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: s_endpgm
define void @merge_global_store_2_constants_i8_natural_align(i8 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i8, i8 addrspace(1)* %out, i32 1

  store i8 123, i8 addrspace(1)* %out.gep.1
  store i8 456, i8 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_2_constants_i16:
; GCN: buffer_store_dword v
define void @merge_global_store_2_constants_i16(i16 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i16, i16 addrspace(1)* %out, i32 1

  store i16 123, i16 addrspace(1)* %out.gep.1
  store i16 456, i16 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_2_constants_0_i16:
; GCN: buffer_store_dword v
define void @merge_global_store_2_constants_0_i16(i16 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i16, i16 addrspace(1)* %out, i32 1

  store i16 0, i16 addrspace(1)* %out.gep.1
  store i16 0, i16 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_2_constants_i16_natural_align:
; GCN: buffer_store_short
; GCN: buffer_store_short
; GCN: s_endpgm
define void @merge_global_store_2_constants_i16_natural_align(i16 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i16, i16 addrspace(1)* %out, i32 1

  store i16 123, i16 addrspace(1)* %out.gep.1
  store i16 456, i16 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_2_constants_i32:
; SI-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 0x1c8
; SI-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0x7b
; GCN: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define void @merge_global_store_2_constants_i32(i32 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1

  store i32 123, i32 addrspace(1)* %out.gep.1
  store i32 456, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_2_constants_i32_f32:
; GCN: buffer_store_dwordx2
define void @merge_global_store_2_constants_i32_f32(i32 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  %out.gep.1.bc = bitcast i32 addrspace(1)* %out.gep.1 to float addrspace(1)*
  store float 1.0, float addrspace(1)* %out.gep.1.bc
  store i32 456, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_2_constants_f32_i32:
; SI-DAG: v_mov_b32_e32 v[[VLO:[0-9]+]], 4.0
; SI-DAG: v_mov_b32_e32 v[[VHI:[0-9]+]], 0x7b
; GCN: buffer_store_dwordx2 v{{\[}}[[VLO]]:[[VHI]]{{\]}}
define void @merge_global_store_2_constants_f32_i32(float addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr float, float addrspace(1)* %out, i32 1
  %out.gep.1.bc = bitcast float addrspace(1)* %out.gep.1 to i32 addrspace(1)*
  store i32 123, i32 addrspace(1)* %out.gep.1.bc
  store float 4.0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_4_constants_i32:
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0x14d{{$}}
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x1c8{{$}}
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x7b{{$}}
; GCN-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 0x4d2{{$}}
; GCN: buffer_store_dwordx4 v{{\[}}[[LO]]:[[HI]]{{\]}}
define void @merge_global_store_4_constants_i32(i32 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  %out.gep.2 = getelementptr i32, i32 addrspace(1)* %out, i32 2
  %out.gep.3 = getelementptr i32, i32 addrspace(1)* %out, i32 3

  store i32 123, i32 addrspace(1)* %out.gep.1
  store i32 456, i32 addrspace(1)* %out.gep.2
  store i32 333, i32 addrspace(1)* %out.gep.3
  store i32 1234, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_4_constants_f32_order:
; GCN: buffer_store_dwordx4
define void @merge_global_store_4_constants_f32_order(float addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr float, float addrspace(1)* %out, i32 1
  %out.gep.2 = getelementptr float, float addrspace(1)* %out, i32 2
  %out.gep.3 = getelementptr float, float addrspace(1)* %out, i32 3

  store float 8.0, float addrspace(1)* %out
  store float 1.0, float addrspace(1)* %out.gep.1
  store float 2.0, float addrspace(1)* %out.gep.2
  store float 4.0, float addrspace(1)* %out.gep.3
  ret void
}

; First store is out of order.
; GCN-LABEL: {{^}}merge_global_store_4_constants_f32:
; GCN: buffer_store_dwordx4
define void @merge_global_store_4_constants_f32(float addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr float, float addrspace(1)* %out, i32 1
  %out.gep.2 = getelementptr float, float addrspace(1)* %out, i32 2
  %out.gep.3 = getelementptr float, float addrspace(1)* %out, i32 3

  store float 1.0, float addrspace(1)* %out.gep.1
  store float 2.0, float addrspace(1)* %out.gep.2
  store float 4.0, float addrspace(1)* %out.gep.3
  store float 8.0, float addrspace(1)* %out
  ret void
}

; FIXME: Should be able to merge this
; GCN-LABEL: {{^}}merge_global_store_4_constants_mixed_i32_f32:
; GCN-NOAA: buffer_store_dword v
; GCN-NOAA: buffer_store_dword v
; GCN-NOAA: buffer_store_dword v
; GCN-NOAA: buffer_store_dword v

; GCN-AA: buffer_store_dwordx2
; GCN-AA: buffer_store_dword v
; GCN-AA: buffer_store_dword v

; GCN: s_endpgm
define void @merge_global_store_4_constants_mixed_i32_f32(float addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr float, float addrspace(1)* %out, i32 1
  %out.gep.2 = getelementptr float, float addrspace(1)* %out, i32 2
  %out.gep.3 = getelementptr float, float addrspace(1)* %out, i32 3

  %out.gep.1.bc = bitcast float addrspace(1)* %out.gep.1 to i32 addrspace(1)*
  %out.gep.3.bc = bitcast float addrspace(1)* %out.gep.3 to i32 addrspace(1)*

  store i32 11, i32 addrspace(1)* %out.gep.1.bc
  store float 2.0, float addrspace(1)* %out.gep.2
  store i32 17, i32 addrspace(1)* %out.gep.3.bc
  store float 8.0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_3_constants_i32:
; SI-DAG: buffer_store_dwordx2
; SI-DAG: buffer_store_dword
; SI-NOT: buffer_store_dword
; GCN: s_endpgm
define void @merge_global_store_3_constants_i32(i32 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  %out.gep.2 = getelementptr i32, i32 addrspace(1)* %out, i32 2

  store i32 123, i32 addrspace(1)* %out.gep.1
  store i32 456, i32 addrspace(1)* %out.gep.2
  store i32 1234, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_2_constants_i64:
; GCN: buffer_store_dwordx4
define void @merge_global_store_2_constants_i64(i64 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i64, i64 addrspace(1)* %out, i64 1

  store i64 123, i64 addrspace(1)* %out.gep.1
  store i64 456, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_4_constants_i64:
; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx4
define void @merge_global_store_4_constants_i64(i64 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i64, i64 addrspace(1)* %out, i64 1
  %out.gep.2 = getelementptr i64, i64 addrspace(1)* %out, i64 2
  %out.gep.3 = getelementptr i64, i64 addrspace(1)* %out, i64 3

  store i64 123, i64 addrspace(1)* %out.gep.1
  store i64 456, i64 addrspace(1)* %out.gep.2
  store i64 333, i64 addrspace(1)* %out.gep.3
  store i64 1234, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_2_adjacent_loads_i32:
; GCN: buffer_load_dwordx2 [[LOAD:v\[[0-9]+:[0-9]+\]]]
; GCN: buffer_store_dwordx2 [[LOAD]]
define void @merge_global_store_2_adjacent_loads_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  %in.gep.1 = getelementptr i32, i32 addrspace(1)* %in, i32 1

  %lo = load i32, i32 addrspace(1)* %in
  %hi = load i32, i32 addrspace(1)* %in.gep.1

  store i32 %lo, i32 addrspace(1)* %out
  store i32 %hi, i32 addrspace(1)* %out.gep.1
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_2_adjacent_loads_i32_nonzero_base:
; GCN: buffer_load_dwordx2 [[LOAD:v\[[0-9]+:[0-9]+\]]], s{{\[[0-9]+:[0-9]+\]}}, 0 offset:8
; GCN: buffer_store_dwordx2 [[LOAD]], s{{\[[0-9]+:[0-9]+\]}}, 0 offset:8
define void @merge_global_store_2_adjacent_loads_i32_nonzero_base(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %in.gep.0 = getelementptr i32, i32 addrspace(1)* %in, i32 2
  %in.gep.1 = getelementptr i32, i32 addrspace(1)* %in, i32 3

  %out.gep.0 = getelementptr i32, i32 addrspace(1)* %out, i32 2
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 3
  %lo = load i32, i32 addrspace(1)* %in.gep.0
  %hi = load i32, i32 addrspace(1)* %in.gep.1

  store i32 %lo, i32 addrspace(1)* %out.gep.0
  store i32 %hi, i32 addrspace(1)* %out.gep.1
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_2_adjacent_loads_shuffle_i32:
; GCN: buffer_load_dword v
; GCN: buffer_load_dword v
; GCN: buffer_store_dword v
; GCN: buffer_store_dword v
define void @merge_global_store_2_adjacent_loads_shuffle_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  %in.gep.1 = getelementptr i32, i32 addrspace(1)* %in, i32 1

  %lo = load i32, i32 addrspace(1)* %in
  %hi = load i32, i32 addrspace(1)* %in.gep.1

  store i32 %hi, i32 addrspace(1)* %out
  store i32 %lo, i32 addrspace(1)* %out.gep.1
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_4_adjacent_loads_i32:
; GCN: buffer_load_dwordx4 [[LOAD:v\[[0-9]+:[0-9]+\]]]
; GCN: buffer_store_dwordx4 [[LOAD]]
define void @merge_global_store_4_adjacent_loads_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  %out.gep.2 = getelementptr i32, i32 addrspace(1)* %out, i32 2
  %out.gep.3 = getelementptr i32, i32 addrspace(1)* %out, i32 3
  %in.gep.1 = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %in.gep.2 = getelementptr i32, i32 addrspace(1)* %in, i32 2
  %in.gep.3 = getelementptr i32, i32 addrspace(1)* %in, i32 3

  %x = load i32, i32 addrspace(1)* %in
  %y = load i32, i32 addrspace(1)* %in.gep.1
  %z = load i32, i32 addrspace(1)* %in.gep.2
  %w = load i32, i32 addrspace(1)* %in.gep.3

  store i32 %x, i32 addrspace(1)* %out
  store i32 %y, i32 addrspace(1)* %out.gep.1
  store i32 %z, i32 addrspace(1)* %out.gep.2
  store i32 %w, i32 addrspace(1)* %out.gep.3
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_3_adjacent_loads_i32:
; SI-DAG: buffer_load_dwordx2
; SI-DAG: buffer_load_dword v
; GCN: s_waitcnt
; SI-DAG: buffer_store_dword v
; SI-DAG: buffer_store_dwordx2 v
; GCN: s_endpgm
define void @merge_global_store_3_adjacent_loads_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  %out.gep.2 = getelementptr i32, i32 addrspace(1)* %out, i32 2
  %in.gep.1 = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %in.gep.2 = getelementptr i32, i32 addrspace(1)* %in, i32 2

  %x = load i32, i32 addrspace(1)* %in
  %y = load i32, i32 addrspace(1)* %in.gep.1
  %z = load i32, i32 addrspace(1)* %in.gep.2

  store i32 %x, i32 addrspace(1)* %out
  store i32 %y, i32 addrspace(1)* %out.gep.1
  store i32 %z, i32 addrspace(1)* %out.gep.2
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_4_adjacent_loads_f32:
; GCN: buffer_load_dwordx4 [[LOAD:v\[[0-9]+:[0-9]+\]]]
; GCN: buffer_store_dwordx4 [[LOAD]]
define void @merge_global_store_4_adjacent_loads_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %out.gep.1 = getelementptr float, float addrspace(1)* %out, i32 1
  %out.gep.2 = getelementptr float, float addrspace(1)* %out, i32 2
  %out.gep.3 = getelementptr float, float addrspace(1)* %out, i32 3
  %in.gep.1 = getelementptr float, float addrspace(1)* %in, i32 1
  %in.gep.2 = getelementptr float, float addrspace(1)* %in, i32 2
  %in.gep.3 = getelementptr float, float addrspace(1)* %in, i32 3

  %x = load float, float addrspace(1)* %in
  %y = load float, float addrspace(1)* %in.gep.1
  %z = load float, float addrspace(1)* %in.gep.2
  %w = load float, float addrspace(1)* %in.gep.3

  store float %x, float addrspace(1)* %out
  store float %y, float addrspace(1)* %out.gep.1
  store float %z, float addrspace(1)* %out.gep.2
  store float %w, float addrspace(1)* %out.gep.3
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_4_adjacent_loads_i32_nonzero_base:
; GCN: buffer_load_dwordx4 [[LOAD:v\[[0-9]+:[0-9]+\]]], s{{\[[0-9]+:[0-9]+\]}}, 0 offset:44
; GCN: buffer_store_dwordx4 [[LOAD]], s{{\[[0-9]+:[0-9]+\]}}, 0 offset:28
define void @merge_global_store_4_adjacent_loads_i32_nonzero_base(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %in.gep.0 = getelementptr i32, i32 addrspace(1)* %in, i32 11
  %in.gep.1 = getelementptr i32, i32 addrspace(1)* %in, i32 12
  %in.gep.2 = getelementptr i32, i32 addrspace(1)* %in, i32 13
  %in.gep.3 = getelementptr i32, i32 addrspace(1)* %in, i32 14
  %out.gep.0 = getelementptr i32, i32 addrspace(1)* %out, i32 7
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 8
  %out.gep.2 = getelementptr i32, i32 addrspace(1)* %out, i32 9
  %out.gep.3 = getelementptr i32, i32 addrspace(1)* %out, i32 10

  %x = load i32, i32 addrspace(1)* %in.gep.0
  %y = load i32, i32 addrspace(1)* %in.gep.1
  %z = load i32, i32 addrspace(1)* %in.gep.2
  %w = load i32, i32 addrspace(1)* %in.gep.3

  store i32 %x, i32 addrspace(1)* %out.gep.0
  store i32 %y, i32 addrspace(1)* %out.gep.1
  store i32 %z, i32 addrspace(1)* %out.gep.2
  store i32 %w, i32 addrspace(1)* %out.gep.3
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_4_adjacent_loads_inverse_i32:
; GCN: buffer_load_dwordx4 [[LOAD:v\[[0-9]+:[0-9]+\]]]
; GCN: s_barrier
; GCN: buffer_store_dwordx4 [[LOAD]]
define void @merge_global_store_4_adjacent_loads_inverse_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  %out.gep.2 = getelementptr i32, i32 addrspace(1)* %out, i32 2
  %out.gep.3 = getelementptr i32, i32 addrspace(1)* %out, i32 3
  %in.gep.1 = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %in.gep.2 = getelementptr i32, i32 addrspace(1)* %in, i32 2
  %in.gep.3 = getelementptr i32, i32 addrspace(1)* %in, i32 3

  %x = load i32, i32 addrspace(1)* %in
  %y = load i32, i32 addrspace(1)* %in.gep.1
  %z = load i32, i32 addrspace(1)* %in.gep.2
  %w = load i32, i32 addrspace(1)* %in.gep.3

  ; Make sure the barrier doesn't stop this
  tail call void @llvm.AMDGPU.barrier.local() #1

  store i32 %w, i32 addrspace(1)* %out.gep.3
  store i32 %z, i32 addrspace(1)* %out.gep.2
  store i32 %y, i32 addrspace(1)* %out.gep.1
  store i32 %x, i32 addrspace(1)* %out

  ret void
}

; TODO: Re-packing of loaded register required. Maybe an IR pass
; should catch this?

; GCN-LABEL: {{^}}merge_global_store_4_adjacent_loads_shuffle_i32:
; GCN: buffer_load_dword v
; GCN: buffer_load_dword v
; GCN: buffer_load_dword v
; GCN: buffer_load_dword v
; GCN: s_barrier
; GCN: buffer_store_dword v
; GCN: buffer_store_dword v
; GCN: buffer_store_dword v
; GCN: buffer_store_dword v
define void @merge_global_store_4_adjacent_loads_shuffle_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  %out.gep.2 = getelementptr i32, i32 addrspace(1)* %out, i32 2
  %out.gep.3 = getelementptr i32, i32 addrspace(1)* %out, i32 3
  %in.gep.1 = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %in.gep.2 = getelementptr i32, i32 addrspace(1)* %in, i32 2
  %in.gep.3 = getelementptr i32, i32 addrspace(1)* %in, i32 3

  %x = load i32, i32 addrspace(1)* %in
  %y = load i32, i32 addrspace(1)* %in.gep.1
  %z = load i32, i32 addrspace(1)* %in.gep.2
  %w = load i32, i32 addrspace(1)* %in.gep.3

  ; Make sure the barrier doesn't stop this
  tail call void @llvm.AMDGPU.barrier.local() #1

  store i32 %w, i32 addrspace(1)* %out
  store i32 %z, i32 addrspace(1)* %out.gep.1
  store i32 %y, i32 addrspace(1)* %out.gep.2
  store i32 %x, i32 addrspace(1)* %out.gep.3

  ret void
}

; GCN-LABEL: {{^}}merge_global_store_4_adjacent_loads_i8:
; GCN: buffer_load_dword [[LOAD:v[0-9]+]]
; GCN: buffer_store_dword [[LOAD]]
; GCN: s_endpgm
define void @merge_global_store_4_adjacent_loads_i8(i8 addrspace(1)* %out, i8 addrspace(1)* %in) #0 {
  %out.gep.1 = getelementptr i8, i8 addrspace(1)* %out, i8 1
  %out.gep.2 = getelementptr i8, i8 addrspace(1)* %out, i8 2
  %out.gep.3 = getelementptr i8, i8 addrspace(1)* %out, i8 3
  %in.gep.1 = getelementptr i8, i8 addrspace(1)* %in, i8 1
  %in.gep.2 = getelementptr i8, i8 addrspace(1)* %in, i8 2
  %in.gep.3 = getelementptr i8, i8 addrspace(1)* %in, i8 3

  %x = load i8, i8 addrspace(1)* %in, align 4
  %y = load i8, i8 addrspace(1)* %in.gep.1
  %z = load i8, i8 addrspace(1)* %in.gep.2
  %w = load i8, i8 addrspace(1)* %in.gep.3

  store i8 %x, i8 addrspace(1)* %out, align 4
  store i8 %y, i8 addrspace(1)* %out.gep.1
  store i8 %z, i8 addrspace(1)* %out.gep.2
  store i8 %w, i8 addrspace(1)* %out.gep.3
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_4_adjacent_loads_i8_natural_align:
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
; GCN: s_endpgm
define void @merge_global_store_4_adjacent_loads_i8_natural_align(i8 addrspace(1)* %out, i8 addrspace(1)* %in) #0 {
  %out.gep.1 = getelementptr i8, i8 addrspace(1)* %out, i8 1
  %out.gep.2 = getelementptr i8, i8 addrspace(1)* %out, i8 2
  %out.gep.3 = getelementptr i8, i8 addrspace(1)* %out, i8 3
  %in.gep.1 = getelementptr i8, i8 addrspace(1)* %in, i8 1
  %in.gep.2 = getelementptr i8, i8 addrspace(1)* %in, i8 2
  %in.gep.3 = getelementptr i8, i8 addrspace(1)* %in, i8 3

  %x = load i8, i8 addrspace(1)* %in
  %y = load i8, i8 addrspace(1)* %in.gep.1
  %z = load i8, i8 addrspace(1)* %in.gep.2
  %w = load i8, i8 addrspace(1)* %in.gep.3

  store i8 %x, i8 addrspace(1)* %out
  store i8 %y, i8 addrspace(1)* %out.gep.1
  store i8 %z, i8 addrspace(1)* %out.gep.2
  store i8 %w, i8 addrspace(1)* %out.gep.3
  ret void
}

; This works once AA is enabled on the subtarget
; GCN-LABEL: {{^}}merge_global_store_4_vector_elts_loads_v4i32:
; GCN: buffer_load_dwordx4 [[LOAD:v\[[0-9]+:[0-9]+\]]]

; GCN-NOAA: buffer_store_dword v
; GCN-NOAA: buffer_store_dword v
; GCN-NOAA: buffer_store_dword v
; GCN-NOAA: buffer_store_dword v

; GCN-AA: buffer_store_dwordx4 [[LOAD]]

; GCN: s_endpgm
define void @merge_global_store_4_vector_elts_loads_v4i32(i32 addrspace(1)* %out, <4 x i32> addrspace(1)* %in) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  %out.gep.2 = getelementptr i32, i32 addrspace(1)* %out, i32 2
  %out.gep.3 = getelementptr i32, i32 addrspace(1)* %out, i32 3
  %vec = load <4 x i32>, <4 x i32> addrspace(1)* %in

  %x = extractelement <4 x i32> %vec, i32 0
  %y = extractelement <4 x i32> %vec, i32 1
  %z = extractelement <4 x i32> %vec, i32 2
  %w = extractelement <4 x i32> %vec, i32 3

  store i32 %x, i32 addrspace(1)* %out
  store i32 %y, i32 addrspace(1)* %out.gep.1
  store i32 %z, i32 addrspace(1)* %out.gep.2
  store i32 %w, i32 addrspace(1)* %out.gep.3
  ret void
}

; GCN-LABEL: {{^}}merge_local_store_2_constants_i8:
; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: s_endpgm
define void @merge_local_store_2_constants_i8(i8 addrspace(3)* %out) #0 {
  %out.gep.1 = getelementptr i8, i8 addrspace(3)* %out, i32 1

  store i8 123, i8 addrspace(3)* %out.gep.1
  store i8 456, i8 addrspace(3)* %out, align 2
  ret void
}

; GCN-LABEL: {{^}}merge_local_store_2_constants_i32:
; GCN-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 0x1c8
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0x7b
; GCN: ds_write2_b32 v{{[0-9]+}}, v[[LO]], v[[HI]] offset1:1{{$}}
define void @merge_local_store_2_constants_i32(i32 addrspace(3)* %out) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(3)* %out, i32 1

  store i32 123, i32 addrspace(3)* %out.gep.1
  store i32 456, i32 addrspace(3)* %out
  ret void
}

; GCN-LABEL: {{^}}merge_local_store_4_constants_i32:
; GCN-DAG: v_mov_b32_e32 [[K2:v[0-9]+]], 0x1c8
; GCN-DAG: v_mov_b32_e32 [[K3:v[0-9]+]], 0x14d
; GCN-DAG: ds_write2_b32 v{{[0-9]+}}, [[K2]], [[K3]] offset0:2 offset1:3

; GCN-DAG: v_mov_b32_e32 [[K0:v[0-9]+]], 0x4d2
; GCN-DAG: v_mov_b32_e32 [[K1:v[0-9]+]], 0x7b
; GCN-DAG: ds_write2_b32 v{{[0-9]+}}, [[K0]], [[K1]] offset1:1

; GCN: s_endpgm
define void @merge_local_store_4_constants_i32(i32 addrspace(3)* %out) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(3)* %out, i32 1
  %out.gep.2 = getelementptr i32, i32 addrspace(3)* %out, i32 2
  %out.gep.3 = getelementptr i32, i32 addrspace(3)* %out, i32 3

  store i32 123, i32 addrspace(3)* %out.gep.1
  store i32 456, i32 addrspace(3)* %out.gep.2
  store i32 333, i32 addrspace(3)* %out.gep.3
  store i32 1234, i32 addrspace(3)* %out
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_5_constants_i32:
; GCN-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 9{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HI4:[0-9]+]], -12{{$}}
; GCN: buffer_store_dwordx4 v{{\[}}[[LO]]:[[HI4]]{{\]}}
; GCN: v_mov_b32_e32 v[[HI:[0-9]+]], 11{{$}}
; GCN: buffer_store_dword v[[HI]]
define void @merge_global_store_5_constants_i32(i32 addrspace(1)* %out) {
  store i32 9, i32 addrspace(1)* %out, align 4
  %idx1 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 1
  store i32 12, i32 addrspace(1)* %idx1, align 4
  %idx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 2
  store i32 16, i32 addrspace(1)* %idx2, align 4
  %idx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 3
  store i32 -12, i32 addrspace(1)* %idx3, align 4
  %idx4 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 4
  store i32 11, i32 addrspace(1)* %idx4, align 4
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_6_constants_i32:
; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx2
define void @merge_global_store_6_constants_i32(i32 addrspace(1)* %out) {
  store i32 13, i32 addrspace(1)* %out, align 4
  %idx1 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 1
  store i32 15, i32 addrspace(1)* %idx1, align 4
  %idx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 2
  store i32 62, i32 addrspace(1)* %idx2, align 4
  %idx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 3
  store i32 63, i32 addrspace(1)* %idx3, align 4
  %idx4 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 4
  store i32 11, i32 addrspace(1)* %idx4, align 4
  %idx5 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 5
  store i32 123, i32 addrspace(1)* %idx5, align 4
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_7_constants_i32:
; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx2
; GCN: buffer_store_dword v
define void @merge_global_store_7_constants_i32(i32 addrspace(1)* %out) {
  store i32 34, i32 addrspace(1)* %out, align 4
  %idx1 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 1
  store i32 999, i32 addrspace(1)* %idx1, align 4
  %idx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 2
  store i32 65, i32 addrspace(1)* %idx2, align 4
  %idx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 3
  store i32 33, i32 addrspace(1)* %idx3, align 4
  %idx4 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 4
  store i32 98, i32 addrspace(1)* %idx4, align 4
  %idx5 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 5
  store i32 91, i32 addrspace(1)* %idx5, align 4
  %idx6 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 6
  store i32 212, i32 addrspace(1)* %idx6, align 4
  ret void
}

; GCN-LABEL: {{^}}merge_global_store_8_constants_i32:
; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx4
; GCN: s_endpgm
define void @merge_global_store_8_constants_i32(i32 addrspace(1)* %out) {
  store i32 34, i32 addrspace(1)* %out, align 4
  %idx1 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 1
  store i32 999, i32 addrspace(1)* %idx1, align 4
  %idx2 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 2
  store i32 65, i32 addrspace(1)* %idx2, align 4
  %idx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 3
  store i32 33, i32 addrspace(1)* %idx3, align 4
  %idx4 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 4
  store i32 98, i32 addrspace(1)* %idx4, align 4
  %idx5 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 5
  store i32 91, i32 addrspace(1)* %idx5, align 4
  %idx6 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 6
  store i32 212, i32 addrspace(1)* %idx6, align 4
  %idx7 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 7
  store i32 999, i32 addrspace(1)* %idx7, align 4
  ret void
}

; This requires handling of scalar_to_vector for v2i64 to avoid
; scratch usage.
; FIXME: Should do single load and store

; GCN-LABEL: {{^}}copy_v3i32_align4:
; GCN-NOT: SCRATCH_RSRC_DWORD
; GCN-DAG: buffer_load_dword v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:8
; GCN-DAG: buffer_load_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
; GCN-NOT: offen
; GCN: s_waitcnt vmcnt
; GCN-NOT: offen
; GCN-DAG: buffer_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
; GCN-DAG: buffer_store_dword v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:8

; GCN: ScratchSize: 0{{$}}
define void @copy_v3i32_align4(<3 x i32> addrspace(1)* noalias %out, <3 x i32> addrspace(1)* noalias %in) #0 {
  %vec = load <3 x i32>, <3 x i32> addrspace(1)* %in, align 4
  store <3 x i32> %vec, <3 x i32> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}copy_v3i64_align4:
; GCN-NOT: SCRATCH_RSRC_DWORD
; GCN-DAG: buffer_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
; GCN-DAG: buffer_load_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:16{{$}}
; GCN-NOT: offen
; GCN: s_waitcnt vmcnt
; GCN-NOT: offen
; GCN-DAG: buffer_store_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
; GCN-DAG: buffer_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:16{{$}}
; GCN: ScratchSize: 0{{$}}
define void @copy_v3i64_align4(<3 x i64> addrspace(1)* noalias %out, <3 x i64> addrspace(1)* noalias %in) #0 {
  %vec = load <3 x i64>, <3 x i64> addrspace(1)* %in, align 4
  store <3 x i64> %vec, <3 x i64> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}copy_v3f32_align4:
; GCN-NOT: SCRATCH_RSRC_DWORD
; GCN-DAG: buffer_load_dword v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:8
; GCN-DAG: buffer_load_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
; GCN-NOT: offen
; GCN: s_waitcnt vmcnt
; GCN-NOT: offen
; GCN-DAG: buffer_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
; GCN-DAG: buffer_store_dword v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:8
; GCN: ScratchSize: 0{{$}}
define void @copy_v3f32_align4(<3 x float> addrspace(1)* noalias %out, <3 x float> addrspace(1)* noalias %in) #0 {
  %vec = load <3 x float>, <3 x float> addrspace(1)* %in, align 4
  %fadd = fadd <3 x float> %vec, <float 1.0, float 2.0, float 4.0>
  store <3 x float> %fadd, <3 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}copy_v3f64_align4:
; GCN-NOT: SCRATCH_RSRC_DWORD
; GCN-DAG: buffer_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
; GCN-DAG: buffer_load_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:16{{$}}
; GCN-NOT: offen
; GCN: s_waitcnt vmcnt
; GCN-NOT: offen
; GCN-DAG: buffer_store_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
; GCN-DAG: buffer_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:16{{$}}
; GCN: ScratchSize: 0{{$}}
define void @copy_v3f64_align4(<3 x double> addrspace(1)* noalias %out, <3 x double> addrspace(1)* noalias %in) #0 {
  %vec = load <3 x double>, <3 x double> addrspace(1)* %in, align 4
  %fadd = fadd <3 x double> %vec, <double 1.0, double 2.0, double 4.0>
  store <3 x double> %fadd, <3 x double> addrspace(1)* %out
  ret void
}

declare void @llvm.AMDGPU.barrier.local() #1

attributes #0 = { nounwind }
attributes #1 = { noduplicate nounwind }
